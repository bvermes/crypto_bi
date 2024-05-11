import sys
sys.path.append('../CRYPTO_BI')
import schedule
import time 
import threading

from business.utils.functions import (
    add_new_pair,
    remove_pair,
    refresh_all_data,
    open_powerbi_dashboard,
    new_unemployment_rate,
    rerun_models,
)

schedule.every(1).minutes.do(refresh_all_data)
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)



def menu():
    quit_menu = False
    while not quit_menu:
        print("Welcome to my Crypto Business Intelligence App!")
        print("Please choose an option from the menu:")
        print("1. Add new pair to the tracker (pair)")
        print("2. Delete an existing pair (pair)")
        print("3. Refresh all the data (run the APIs for all the endpoints)")
        print("4. Open PowerBI dashboard")
        print("5. Update/insert monthly unemployment (month, percentage)")
        print("6. Rerun models")
        print("Q. Quit the menu")

        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            pair = input(
                "First give me the name of the pair you want to add to the tracker.(example: BTCUSDT):"
            )
            exchange = input("Now the exchange of the pair.(example: BINANCE):")
            add_new_pair(pair, exchange)
        elif choice == "2":
            pair = input(
                "First give me the name of the pair you want to remove(example: BTCUSDT):"
            )
            exchange = input("Now the exchange of the pair.(example: BINANCE):")
            remove_pair(pair, exchange)
        elif choice == "3":
            refresh_all_data()
        elif choice == "4":
            open_powerbi_dashboard()
        elif choice == "5":
            month = input("Please write the month(example: 2023/12):")
            rate = input("Please write the unemployment_rate(example: 3.4):")
            new_unemployment_rate(month, rate)
        elif choice == "6":
            refresh_all_data()
            pair = input("What pair do you want to use to traing the model? (example: BTCUSDT):")
            rerun_models(pair)
            pass
        elif choice.upper() == "Q":
            print("Program closing. Goodbye!")
            quit_menu = True
        else:
            print("Invalid choice. Please enter a number from 1 to 6, or Q to quit.")


if __name__ == "__main__":
    refresh_all_data()
    #etl_thread = threading.Thread(target=run_scheduler)
    #etl_thread.start()
    menu()
