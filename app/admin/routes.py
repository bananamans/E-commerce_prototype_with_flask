from flask import Blueprint, render_template, url_for, flash
from werkzeug.utils import redirect
from ..db_models import Order, Ordered_item, Item, db
from ..admin.forms import AddItemForm, EditItemForm, OrderEditForm
from ..funcs import admin_only

admin = Blueprint("admin", __name__, url_prefix="/admin", static_folder="static", template_folder="templates")

@admin.route('/')
# @admin_only
def dashboard():
    items = Item.query.all()
    return render_template("admin/items.html", items=items)

@admin.route('/items')
# @admin_only
def items():
    items = Item.query.all()
    return render_template("admin/items.html", items=items)

@admin.route('/add', methods=['POST', 'GET'])
# @admin_only
def add():
    form = AddItemForm()

    if form.validate_on_submit():
        name = form.name.data
        price = form.price.data
        category = form.category.data
        brand = form.brand.data
        details = form.details.data
        total_rating = 5;
        num_rating = 1;
        form.image.data.save('app/static/uploads/' + form.image.data.filename)
        image = url_for('static', filename=f'uploads/{form.image.data.filename}')
        item = Item(name=name, price=price, category=category, brand=brand, details=details, total_rating=total_rating, num_rating=num_rating, image=image)
        db.session.add(item)
        db.session.commit()
        flash(f'{name} added successfully!','success')
        return redirect(url_for('admin.items'))
    return render_template("admin/add.html", form=form)

@admin.route('/edit/<string:type>/<int:id>', methods=['POST', 'GET'])
# @admin_only
def edit(type, id):
    if type == "item":
        item = Item.query.get(id)
        form = EditItemForm(
            name = item.name,
            price = item.price,
            category = item.category,
            brand = item.brand,
            details = item.details,
        )
        if form.validate_on_submit():
            item.name = form.name.data
            item.price = form.price.data
            item.category = form.category.data
            item.brand = form.brand.data
            item.details = form.details.data
            db.session.commit()
            return redirect(url_for('admin.items'))
    elif type == "order":
        order = Order.query.get(id)
        form = OrderEditForm(status = order.status)
        if form.validate_on_submit():
            order.status = form.status.data
            db.session.commit()
            return redirect(url_for('admin.dashboard'))
    return render_template('admin/add.html', form=form)

@admin.route('/delete/<int:id>')
# @admin_only
def delete(id):
    Ordered_item.query.filter_by(itemid=id).delete()
    to_delete = Item.query.get(id)
    db.session.delete(to_delete)
    db.session.commit()
    flash(f'{to_delete.name} deleted successfully', 'error')
    return redirect(url_for('admin.items'))