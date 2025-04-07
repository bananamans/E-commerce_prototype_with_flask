import datetime
from flask_login import current_user
from dotenv import load_dotenv
from .db_models import Order, Ordered_item, db, User


load_dotenv()

def fulfill_order(session):
	""" Fulfils order on successful payment """

	uid = session['client_reference_id']
	order = Order(uid=uid, date=datetime.datetime.now(), status="processing")
	db.session.add(order)
	db.session.commit()

	current_user = User.query.get(uid)
	for cart in current_user.cart:
		ordered_item = Ordered_item(oid=order.id, itemid=cart.item.id, quantity=cart.quantity)
		db.session.add(ordered_item)
		db.session.commit()
		current_user.remove_from_cart(cart.item.id, cart.quantity)
		db.session.commit()
		