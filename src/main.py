
"""
Instamart AI 360
AI-Powered Customer Intelligence Platform
"""

def display_welcome():

    print("=" * 55)
    print("       INSTAMART AI 360")
    print("  CUSTOMER INTELLIGENCE PLATFORM")
    print("=" * 55)


def display_modules():

    print("\nAvailable Modules:")

    modules = [
        "Customer Data Intelligence",
        "Customer Segmentation",
        "Churn Prediction",
        "Smart Ticket Classification",
        "RAG Customer Support Copilot",
        "Proactive Customer Intelligence"
    ]

    for index, module in enumerate(modules, start=1):
        print(f"{index}. {module}")


def main():

    display_welcome()
    display_modules()

    print("\nProject initialized successfully!")
    print("Welcome to Instamart AI 360!")


if __name__ == "__main__":
    main()
