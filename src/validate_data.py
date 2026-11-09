
from pathlib import Path
import pandas as pd

# Dataset folder
DATA_DIR = Path("data/synthetic")

# Load datasets
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
order_items = pd.read_csv(DATA_DIR / "order_items.csv")
tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")

print("\n========== DATASET COUNTS ==========")

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Order Items:", len(order_items))
print("Support Tickets:", len(tickets))


# Check primary keys
print("\n========== PRIMARY KEY VALIDATION ==========")

assert customers["customer_id"].is_unique
assert products["product_id"].is_unique
assert orders["order_id"].is_unique
assert order_items["order_item_id"].is_unique
assert tickets["ticket_id"].is_unique

print("All primary keys are unique!")


# Check foreign keys
print("\n========== FOREIGN KEY VALIDATION ==========")

assert orders["customer_id"].isin(
    customers["customer_id"]
).all()

assert order_items["order_id"].isin(
    orders["order_id"]
).all()

assert order_items["product_id"].isin(
    products["product_id"]
).all()

assert tickets["customer_id"].isin(
    customers["customer_id"]
).all()

assert tickets["order_id"].isin(
    orders["order_id"]
).all()

print("All foreign key relationships are valid!")


# Final result
print("\n========== VALIDATION RESULT ==========")
print("SUCCESS: All datasets passed validation!")
