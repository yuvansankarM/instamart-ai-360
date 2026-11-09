
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# INSTAMART AI 360 - DATASET GENERATOR
# ==========================================

rng = np.random.default_rng(42)

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = pd.Timestamp.today().normalize()

# ==========================================
# 1. GENERATE CUSTOMERS
# ==========================================

customer_count = 500

cities = [
    "Chennai", "Salem", "Coimbatore",
    "Erode", "Bengaluru", "Hyderabad",
    "Mumbai", "Pune"
]

customers = pd.DataFrame({
    "customer_id": [
        f"CUST{i:04d}" for i in range(1, customer_count + 1)
    ],
    "name": [
        f"Customer_{i:04d}" for i in range(1, customer_count + 1)
    ],
    "age": rng.integers(18, 71, customer_count),
    "city": rng.choice(cities, customer_count),
    "signup_date": (
        TODAY - pd.to_timedelta(
            rng.integers(30, 366, customer_count), unit="D"
        )
    ),
    "membership": rng.choice(
        ["Regular", "Silver", "Gold"],
        customer_count,
        p=[0.55, 0.30, 0.15]
    )
})

# ==========================================
# 2. GENERATE PRODUCTS
# ==========================================

catalogue = {
    "Dairy": ["Milk", "Curd", "Butter", "Cheese"],
    "Fruits": ["Apple", "Banana", "Orange", "Grapes"],
    "Vegetables": ["Tomato", "Potato", "Onion", "Carrot"],
    "Snacks": ["Chips", "Biscuits", "Chocolate", "Popcorn"],
    "Beverages": ["Juice", "Coffee", "Tea", "Soft Drink"],
    "Personal Care": ["Soap", "Shampoo", "Toothpaste"],
    "Household": ["Detergent", "Dishwash", "Tissue Paper"]
}

products = []

for i in range(1, 81):
    category = rng.choice(list(catalogue.keys()))
    product_name = rng.choice(catalogue[category])

    products.append({
        "product_id": f"PROD{i:04d}",
        "product_name": f"{product_name} {i}",
        "category": category,
        "price": round(float(rng.uniform(20, 500)), 2),
        "stock_status": rng.choice(
            ["In Stock", "Low Stock", "Out of Stock"],
            p=[0.75, 0.18, 0.07]
        )
    })

products = pd.DataFrame(products)

# ==========================================
# 3. GENERATE ORDERS AND ORDER ITEMS
# ==========================================

orders = []
order_items = []

order_number = 1
item_number = 1

for _, customer in customers.iterrows():

    # Simulate different purchasing frequencies.
    inactive = rng.random() < 0.30

    if inactive:
        number_of_orders = max(1, int(rng.poisson(1.5)))
    else:
        number_of_orders = max(1, int(rng.poisson(5)))

    earliest_date = max(
        pd.Timestamp(customer["signup_date"]),
        TODAY - pd.Timedelta(days=180)
    )

    available_days = max(
        1, (TODAY - earliest_date).days + 1
    )

    for _ in range(number_of_orders):

        random_day = int(rng.integers(0, available_days))

        order_date = earliest_date + pd.Timedelta(days=random_day)

        order_date += pd.Timedelta(
            minutes=int(rng.integers(0, 1440))
        )

        order_id = f"ORD{order_number:05d}"

        order_status = rng.choice(
            ["Delivered", "Delayed", "Cancelled"],
            p=[0.84, 0.11, 0.05]
        )

        delivery_minutes = (
            int(rng.integers(15, 36))
            if order_status == "Delivered"
            else (
                int(rng.integers(36, 90))
                if order_status == "Delayed"
                else None
            )
        )

        orders.append({
            "order_id": order_id,
            "customer_id": customer["customer_id"],
            "order_date": order_date,
            "order_status": order_status,
            "delivery_minutes": delivery_minutes,
            "payment_method": rng.choice(
                ["UPI", "Card", "Cash", "Wallet"]
            ),
            "discount_amount": round(
                float(rng.choice([0, 10, 20, 30, 50])), 2
            )
        })

        # One order can contain multiple products.
        number_of_items = int(rng.integers(1, 6))

        selected_products = rng.choice(
            len(products),
            size=min(number_of_items, len(products)),
            replace=False
        )

        for product_index in selected_products:

            product = products.iloc[product_index]

            quantity = int(rng.integers(1, 4))

            order_items.append({
                "order_item_id": f"ITEM{item_number:06d}",
                "order_id": order_id,
                "product_id": product["product_id"],
                "quantity": quantity,
                "unit_price": product["price"]
            })

            item_number += 1

        order_number += 1

orders = pd.DataFrame(orders)
order_items = pd.DataFrame(order_items)

# Calculate order totals from actual item details.
item_totals = (
    order_items.assign(
        line_total=order_items["quantity"] * order_items["unit_price"]
    )
    .groupby("order_id")["line_total"]
    .sum()
)

orders["order_value"] = (
    orders["order_id"].map(item_totals).round(2)
)

# ==========================================
# 4. GENERATE SUPPORT TICKETS
# ==========================================

ticket_templates = {
    "Delivery": [
        ("My order arrived late.", "Negative", "High"),
        ("Where is my order? It is delayed.", "Negative", "High"),
        ("The delivery experience was good.", "Positive", "Low")
    ],
    "Refund": [
        ("I have not received my refund.", "Negative", "High"),
        ("Please check my refund status.", "Neutral", "Medium"),
        ("Thank you for processing my refund.", "Positive", "Low")
    ],
    "Product Quality": [
        ("The product I received was damaged.", "Negative", "High"),
        ("The item quality was poor.", "Negative", "Medium"),
        ("The product quality was excellent.", "Positive", "Low")
    ],
    "Payment": [
        ("My payment was deducted twice.", "Negative", "High"),
        ("Payment failed but money was deducted.", "Negative", "High"),
        ("I need help with my payment.", "Neutral", "Medium")
    ],
    "Account": [
        ("I cannot log into my account.", "Negative", "Medium"),
        ("Please help me update my account details.", "Neutral", "Low"),
        ("Thank you for your support.", "Positive", "Low")
    ]
}

tickets = []

for i in range(1, 601):

    selected_order = orders.iloc[
        int(rng.integers(0, len(orders)))
    ]

    category = rng.choice(list(ticket_templates.keys()))

    text, sentiment, priority = ticket_templates[category][
        int(rng.integers(0, len(ticket_templates[category])))
    ]

    ticket_date = (
        pd.Timestamp(selected_order["order_date"])
        + pd.Timedelta(days=int(rng.integers(0, 6)))
    )

    ticket_date = min(ticket_date, TODAY)

    tickets.append({
        "ticket_id": f"TKT{i:05d}",
        "customer_id": selected_order["customer_id"],
        "order_id": selected_order["order_id"],
        "ticket_date": ticket_date,
        "ticket_text": text,
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
        "ticket_status": rng.choice(
            ["Open", "In Progress", "Resolved"],
            p=[0.20, 0.25, 0.55]
        )
    })

support_tickets = pd.DataFrame(tickets)

# ==========================================
# 5. SAVE ALL DATASETS
# ==========================================

datasets = {
    "customers.csv": customers,
    "products.csv": products,
    "orders.csv": orders,
    "order_items.csv": order_items,
    "support_tickets.csv": support_tickets
}

for filename, dataframe in datasets.items():

    filepath = OUTPUT_DIR / filename

    dataframe.to_csv(filepath, index=False)

    print(f"Saved: {filepath}")
    print(f"Records: {len(dataframe)}")
    print("-" * 40)

print("\nAll synthetic datasets generated successfully!")
