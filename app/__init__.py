from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, redirect, url_for, flash, request, abort
from flask_bootstrap import Bootstrap
from .forms import LoginForm, RegisterForm
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
from .db_models import db, User, Item, Cart, Order, Ordered_item
from dotenv import load_dotenv
import tensorflow as tf
from .admin.routes import admin
from recommenders.models.ncf.ncf_singlenode import NCF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import OrderedDict

load_dotenv()
app = Flask(__name__)
app.register_blueprint(admin)

app.config["SECRET_KEY"] = "josh_fyp"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///test.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_USERNAME'] = ""
app.config['MAIL_PASSWORD'] = ""
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_PORT'] = 587
# stripe.api_key = ""

Bootstrap(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)

with app.app_context():
	db.create_all()

@app.context_processor
def inject_now():
	""" sends datetime to templates as 'now' """
	return {'now': datetime.utcnow()}

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(user_id)

# Loading model
ncfmodel = NCF(
    n_users= 5284,
    n_items= 34,
    model_type="NeuMF",
    n_factors=12,
    layer_sizes=[8],
    n_epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    verbose=10,
    seed=42
)

ncfmodel.load(neumf_dir = 'app/ncf_model')

def recommend(user_id, all_items):
    TOP_K = 30
    """Filter items using CBF before passing them to NCF."""
    past_orders = Order.query.filter_by(uid=user_id).all()
    ordered_items = [ordered_item.item for order in past_orders for ordered_item in order.items]
    if not ordered_items:
        return []
	
    reference_item = ordered_items[-1]  # Last purchased item
    category, brand = reference_item.category, reference_item.brand

    # Apply CBF (Cosine Similarity)
    cbf_filtered_item_ids = [item.id for item in all_items]
    df = pd.DataFrame([(item.id, item.category, item.brand) for item in all_items], 
                      columns=["itemID", "category", "brand"])
	
    combined_features = df["category"] + " " + df["brand"]
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors, feature_vectors)

    target_idx = df[(df["category"] == category) & (df["brand"] == brand)].index
    if target_idx.empty:
         target_idx = df[(df["category"] == category) | (df["brand"] == brand)].index
	
    if len(target_idx):
        sim_scores = list(enumerate(similarity[target_idx[0]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

		# Exclude the target item itself and get the top K similar items
        top_k_similar_indices = [i[0] for i in sim_scores[1:TOP_K+1]]
        cbf_filtered_item_ids = df.iloc[top_k_similar_indices]["itemID"].tolist()
        # cbf_filtered_item_ids = [df.iloc[i[0]]["itemID"] for i in sim_scores]

    cbf_filtered_items = Item.query.filter(Item.id.in_(cbf_filtered_item_ids)).all()

	# Apply ncf
    ncfmodel.user2id = create_user_to_id_mapping()
    ncfmodel.item2id = create_item_to_id_mapping()

    predictions = [(item, ncfmodel.predict(user_id, item.id)) for item in cbf_filtered_items]
    sorted_recommendations = [item for item, _ in sorted(predictions, key=lambda x: x[1], reverse=True)]
    
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    for i, (item, score) in enumerate(sorted_predictions, start=1):
        print(f"{i}. {item.name} (ID: {item.id}) — Predicted Rating: {score:.4f}")
	
    return sorted_recommendations

def create_item_to_id_mapping():
    """Creates an OrderedDict mapping item IDs to a sequential integer ID."""
    items = Item.query.order_by(Item.id).all()
    item_to_id = OrderedDict()
    for i, item in enumerate(items):
        item_to_id[item.id] = i
    return item_to_id

def create_user_to_id_mapping():
    """Creates an OrderedDict mapping user IDs to a sequential integer ID."""
    users = User.query.order_by(User.id).all()
    user_to_id = OrderedDict()
    for i, user in enumerate(users):
        user_to_id[user.id] = i
    return user_to_id

@app.route("/")
def home():
    items = Item.query.all()
    recommendations = items

    if current_user.is_authenticated:
        recommended_items = recommend(current_user.id, items)
        if recommended_items:
            recommendations = recommended_items  # Show personalized recommendations

    return render_template("home.html", items=recommendations)

@app.route("/login", methods=['POST', 'GET'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		email = form.email.data
		user = User.query.filter_by(email=email).first()
		if user == None:
			flash(f'User with email {email} doesn\'t exist!<br> <a href={url_for("register")}>Register now!</a>', 'error')
			return redirect(url_for('login'))
		elif check_password_hash(user.password, form.password.data):
			login_user(user)
			return redirect(url_for('home'))
		else:
			flash("Email and password incorrect!!", "error")
			return redirect(url_for('login'))
	return render_template("login.html", form=form)

@app.route("/register", methods=['POST', 'GET'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegisterForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user:
			flash(f"User with email {user.email} already exists!!<br> <a href={url_for('login')}>Login now!</a>", "error")
			return redirect(url_for('register'))
		new_user = User(name=form.name.data,
						email=form.email.data,
						password=generate_password_hash(
									form.password.data,
									method='pbkdf2:sha256',
									salt_length=8),
						phone=form.phone.data)
		db.session.add(new_user)
		db.session.commit()
		# send_confirmation_email(new_user.email)
		flash('Thanks for registering! You may login now.', 'success')
		return redirect(url_for('login'))
	return render_template("register.html", form=form)

@app.route("/logout")
@login_required
def logout():
	logout_user()
	return redirect(url_for('login'))

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_id = request.form['admin_id']
        password = request.form['password']
        
        if admin_id == 'admin' and password == 'admin123':
            return redirect(url_for('admin.dashboard')) 
        else:
            flash("Invalid Admin ID or Password", "error")
    
    return render_template('admin_login.html')

@app.route("/add/<id>", methods=['POST'])
def add_to_cart(id):
	if not current_user.is_authenticated:
		flash(f'You must login first!<br> <a href={url_for("login")}>Login now!</a>', 'error')
		return redirect(url_for('login'))

	item = Item.query.get(id)
	if request.method == "POST":
		quantity = request.form["quantity"]
		current_user.add_to_cart(id, quantity)
		flash(f'''{item.name} successfully added to the <a href=cart>cart</a>.<br> <a href={url_for("cart")}>view cart!</a>''','success')
		return redirect(url_for('home'))

@app.route("/cart")
@login_required
def cart():
	price = 0
	items = []
	quantity = []
	for cart in current_user.cart:
		items.append(cart.item)
		quantity.append(cart.quantity)
		price += cart.item.price*cart.quantity
	return render_template('cart.html', items=items, price=price, quantity=quantity)

@app.route('/orders')
@login_required
def orders():
	return render_template('orders.html', orders=current_user.orders)

@app.route('/give_rating/<int:order_id>/<int:user_id>', methods=['GET', 'POST'])
@login_required
def give_rating(order_id, user_id):
    # Retrieve the order and check if the current user is the owner of the order
    order = Order.query.get_or_404(order_id)

    # Ensure that the order belongs to the user
    if order.uid != user_id:
        abort(403)  # Optionally, handle unauthorized access
    
    # Check if the order has already been rated
    if order.rated == 1:
        flash("You have already given a rating for this order.", 'info')
        return render_template('give_rating.html', order=order, already_rated=True)
    
    # If the form is submitted, process the ratings for each ordered item
    if request.method == 'POST':
        for ordered_item in order.items:
            rating_key = f"rating_{ordered_item.id}"
            rating = request.form.get(rating_key)
			
        # Loop through each ordered item and update the ratings
        for ordered_item in order.items:
            rating_key = f"rating_{ordered_item.id}"
            rating = request.form.get(rating_key)
            
            if rating:  # If a rating has been provided for this item
                rating = float(rating)  # Convert rating to float

                # Update the total rating and num_rating of the item
                item = Item.query.get(ordered_item.itemid)

                if item.total_rating is None:
                    item.total_rating = 0
                    item.num_rating = 0

                item.total_rating += rating
                item.num_rating += 1
                db.session.commit()

        # After updating ratings, mark the order as rated
        order.rated = 1
        db.session.commit()

        flash("Your ratings have been submitted!", 'success')
        return redirect(url_for('orders'))

    # If the order hasn't been rated yet, display the rating form
    return render_template('give_rating.html', order=order, already_rated=False)

@app.route("/remove/<id>/<quantity>")
@login_required
def remove(id, quantity):
	current_user.remove_from_cart(id, quantity)
	return redirect(url_for('cart'))

@app.route('/item/<int:id>')
def item(id):
	item = Item.query.get(id)
	return render_template('item.html', item=item)

@app.route('/search')
def search():
	query = request.args['query']
	search = "%{}%".format(query)
	items = Item.query.filter(Item.name.like(search)).all()
	return render_template('home.html', items=items, search=True, query=query)

# stripe stuffs
@app.route('/payment_success')
def payment_success():
	return render_template('success.html')

@app.route('/payment_failure')
def payment_failure():
	return render_template('failure.html')

@app.route('/checkout', methods=['POST'])
def checkout():
	if not current_user.is_authenticated:
		return redirect(url_for('login'))
	
	cart_items = Cart.query.filter_by(uid=current_user.id).all()

	if not cart_items:
		return redirect(url_for('home'))
	
	malaysia_tz = timezone(timedelta(hours=8))
	new_order = Order(
        uid=current_user.id,
        date=datetime.now(malaysia_tz),
        status="Completed",
		rated = 0
    )
	db.session.add(new_order)
	db.session.commit()

	for cart_item in cart_items:
		ordered_item = Ordered_item(
            oid=new_order.id,
            itemid=cart_item.itemid,
            quantity=cart_item.quantity
        )
		db.session.add(ordered_item)
		db.session.delete(cart_item)

	db.session.commit()
	
	return redirect(url_for('home'))