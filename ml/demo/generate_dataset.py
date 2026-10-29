import numpy as np
import pandas as pd
import os

def generate_rul_dataset():
    np.random.seed(42)
    n_samples = 1000

    # 1. Identifier
    record_ids = [f"RUL{i:04d}" for i in range(1, n_samples + 1)]

    # 2. Base Numerical Predictors with realistic correlations
    # Operating Hours: 200 to 5000 hours
    operating_hours = np.random.uniform(250, 4800, n_samples)

    # Ambient Temperature: 18 to 38 °C
    ambient_temp = np.random.normal(28, 4.5, n_samples)
    ambient_temp = np.clip(ambient_temp, 16.0, 40.0)

    # Load: 45% to 95%
    load = np.random.uniform(45.0, 95.0, n_samples)

    # Rotational Speed: Correlated with Load (mean ~ 1800 rpm)
    rotational_speed = 1500.0 + 4.5 * load + np.random.normal(0, 45, n_samples)

    # Pressure: Correlated with Load (mean ~ 110 bar)
    pressure = 65.0 + 0.65 * load + np.random.normal(0, 5.0, n_samples)

    # Temperature: Correlated with Ambient Temp, Load, and Operating Hours
    temperature = (
        25.0
        + 0.9 * ambient_temp
        + 0.38 * load
        + 0.0035 * operating_hours
        + np.random.normal(0, 2.5, n_samples)
    )

    # Vibration: Correlated with Rotational Speed, Operating Hours, and Load
    vibration = (
        0.8
        + 0.0009 * rotational_speed
        + 0.00045 * operating_hours
        + 0.012 * load
        + np.random.normal(0, 0.25, n_samples)
    )

    # Maintenance Count: Integer counts, correlated with Operating Hours
    maintenance_count = np.random.poisson(lam=operating_hours / 1100.0)
    maintenance_count = np.clip(maintenance_count, 0, 7)

    # 3. Categorical Predictors
    machine_types = np.random.choice(['Type A', 'Type B', 'Type C'], size=n_samples, p=[0.42, 0.35, 0.23])
    shifts = np.random.choice(['Morning', 'Evening', 'Night'], size=n_samples, p=[0.35, 0.35, 0.30])

    # Category effects on RUL
    type_map = {'Type A': 0.0, 'Type B': 140.0, 'Type C': -90.0}
    type_effect = np.array([type_map[t] for t in machine_types])

    shift_map = {'Morning': 25.0, 'Evening': 0.0, 'Night': -35.0}
    shift_effect = np.array([shift_map[s] for s in shifts])

    # 4. Target: Remaining Useful Life Hours
    # Continuous positive regression target depending on predictors + interaction + noise
    rul = (
        3300.0
        - 0.28 * operating_hours
        - 13.5 * (temperature - 55.0)
        - 120.0 * (vibration - 2.0)
        - 8.0 * (load - 50.0)
        + 65.0 * maintenance_count
        + type_effect
        + shift_effect
        - 0.04 * (temperature - 60.0) * (vibration - 2.5)  # non-linear interaction
        + np.random.normal(0, 35.0, n_samples)
    )
    # Ensure strictly positive continuous target
    rul = np.clip(rul, 65.0, 3200.0)
    rul = np.round(rul, 2)

    # Round numerical predictors to realistic precision before adding outliers/nulls
    operating_hours = np.round(operating_hours, 1)
    temperature = np.round(temperature, 2)
    vibration = np.round(vibration, 3)
    pressure = np.round(pressure, 2)
    rotational_speed = np.round(rotational_speed, 1)
    load = np.round(load, 2)
    ambient_temp = np.round(ambient_temp, 2)

    # 5. Inject a few mild outliers
    # Mild outliers in Temperature (e.g., 6 observations)
    temp_outlier_idx = [45, 182, 360, 523, 741, 895]
    temperature[temp_outlier_idx] += np.array([24.0, 28.5, 26.0, 31.0, 27.5, 29.0])

    # Mild outliers in Vibration (e.g., 6 observations)
    vib_outlier_idx = [78, 215, 412, 630, 804, 955]
    vibration[vib_outlier_idx] += np.array([3.8, 4.2, 3.5, 4.5, 3.9, 4.1])

    # Mild outliers in Pressure (e.g., 5 observations)
    press_outlier_idx = [102, 310, 580, 720, 888]
    pressure[press_outlier_idx] += np.array([45.0, 52.0, 48.0, 55.0, 50.0])

    # Build DataFrame
    df = pd.DataFrame({
        'Record ID': record_ids,
        'Operating Hours': operating_hours,
        'Temperature': temperature,
        'Vibration': vibration,
        'Pressure': pressure,
        'Rotational Speed': rotational_speed,
        'Load': load,
        'Maintenance Count': maintenance_count,
        'Ambient Temperature': ambient_temp,
        'Machine Type': machine_types.astype(object),
        'Shift': shifts.astype(object),
        'Remaining Useful Life Hours': rul
    })

    # 6. Inject missing values into selected predictors
    # Numerical columns missing values
    num_missing_specs = {
        'Operating Hours': 12,
        'Temperature': 18,
        'Vibration': 15,
        'Pressure': 20,
        'Ambient Temperature': 14
    }
    for col, count in num_missing_specs.items():
        missing_indices = np.random.choice(n_samples, size=count, replace=False)
        df.loc[missing_indices, col] = np.nan

    # Categorical columns missing values
    cat_missing_specs = {
        'Machine Type': 15,
        'Shift': 12
    }
    for col, count in cat_missing_specs.items():
        missing_indices = np.random.choice(n_samples, size=count, replace=False)
        df.loc[missing_indices, col] = np.nan

    # Output path
    output_dir = r"c:\Directory\coding\3122247001061\mltest\ml\demo"
    output_path = os.path.join(output_dir, "machine_rul.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_rul_dataset()

