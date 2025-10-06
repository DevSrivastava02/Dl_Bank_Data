import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Create dummy churn dataset
# -------------------------------
np.random.seed(42)
data = pd.DataFrame({
    "RollNumber": range(1, 11),
    "CustomerID": range(1, 11),
    "Surname": ["Smith", "Johnson", "Williams", "Brown", "Jones",
                "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"],
    "Geography": np.random.choice(["France", "Spain", "Germany"], size=10),
    "Gender": np.random.choice(["Male", "Female"], size=10),
    "Age": np.random.randint(25, 60, size=10),
    "Tenure": np.random.randint(1, 10, size=10),
    "Balance": np.random.uniform(0, 250000, size=10),
    "NumOfProducts": np.random.randint(1, 4, size=10),
    "HasCrCard": np.random.randint(0, 2, size=10),
    "IsActiveMember": np.random.randint(0, 2, size=10),
    "EstimatedSalary": np.random.uniform(5000, 150000, size=10),
    "Exited": np.random.randint(0, 2, size=10)
})



# -------------------------------
# Step 3: Save dataset using open()
# -------------------------------
with open("chunk_data.csv", "w") as file:
    data.to_csv(file, index=False)

print("chunk_data.csv created successfully!")
print(data.head())
