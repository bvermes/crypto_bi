import subprocess
import os


def open_power_bi_dashboard():
    POWER_BI_DESKTOP_PATH = r"C:\Users\Balazs\AppData\Local\Microsoft\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_8wekyb3d8bbwe\PBIDesktopStore.exe"
    try:
        dashboard_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "dashboard.pbix"
        )

        subprocess.Popen([POWER_BI_DESKTOP_PATH, dashboard_path])
        print("Power BI Desktop opened successfully.")
    except FileNotFoundError:
        print(
            "Power BI Desktop not found. Please check the path to Power BI Desktop executable."
        )
