import os


def create_test_files() -> None:
    """
    Creates a folder named 'test_files' and generates test files with predefined contents.
    """
    # Create the test_files directory if it doesn't exist
    os.makedirs("test_files", exist_ok=True)

    files_content = {
        "test_files/test_data_normal.txt": """Temp    Density
3405 58.05
3300 57.0
3250 56.5
3100 55.0
3000 54.0
2905 53.05
""",
        "test_files/test_data_normal1.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3050 54.5
3000 54.0
2900 53.0
2800 52.0
""",
        "test_files/test_data_variable_spacing.txt": """Temp    Density
3400 58.0
3250 56.5
3150 55.5
3050 54.5
2950 53.5
2850 52.5
2750 51.5
2650 50.5
""",
        "test_files/test_data_sparse.txt": """Temp    Density
3400 58.0
3000 54.0
2600 50.0
2200 46.0
""",
        "test_files/test_data_missing_values.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3000 NaN
2900 53.0
2800 52.1
""",
        "test_files/test_data_descending.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3000 54.0
2900 53.0
2800 52.1
""",
        "test_files/test_data_duplicates.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3050 54.5
3000 54.0
3000 53.5
2900 53.0
""",
        "test_files/test_data_empty.txt": """
""",
        "test_files/test_data_header.txt": """Temp    Density

""",
        "test_files/test_data_invalid_data.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3000 Invalid
2900 53.0
2800 52.1
""",
        "test_files/test_data_mismatched_columns.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0 1
3100 55.0
3050 54.5 2 3
3000 54.0
3000 53.5
2900 53.0 4
""",
        "test_files/test_data_missing_data.txt": """Temp    Density
3400 58.0
3300 57.0
3200 56.0
3100 55.0
3050 
3000 54.0
3000 53.5
2900 53
""",
        "test_files/test_data_density_temperature_ascending.txt": """Temp    Density
1500.0	7.008098755
1490.0	7.015835416
1480.0	7.023544996
1470.0	7.031227258
1460.0	7.038881969
1459.63	7.039161939
1450.0	7.101066042
1440.0	7.134387853
1434.71	7.147271626
1430.0	7.177458788
1420.0	7.211193546
1410.0	7.228319303
1400.0	7.239681101
""",
        "test_files/test_data_density_temperature_descending.txt": """Temp    Density
1500.0	7.008098755
1490.0	7.015835416
1480.0	7.023544996
1470.0	7.031227258
1460.0	7.038881969
1459.63	7.039161939
1450.0	7.101066042
1440.0	7.134387853
1434.71	7.147271626
1430.0	7.177458788
1420.0	7.211193546
1410.0	7.228319303
1400.0	7.239681101
""",
    }

    for filename, content in files_content.items():
        with open(filename, 'w') as file:
            file.write(content)
