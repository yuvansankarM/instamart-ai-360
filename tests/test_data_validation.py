from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"


def test_customers_dataset_exists():
    assert (DATA_DIR / "customers.csv").exists()


def test_orders_dataset_exists():
    assert (DATA_DIR / "orders.csv").exists()


def test_customers_primary_key():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    assert customers["customer_id"].is_unique


def test_orders_primary_key():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    assert orders["order_id"].is_unique


def test_orders_customer_relationship():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    assert orders["customer_id"].isin(customers["customer_id"]).all()


def test_customer_dataset_not_empty():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    assert len(customers) > 0
