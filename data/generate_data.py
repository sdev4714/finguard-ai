import pandas as pd
import numpy as np

np.random.seed(42)

def generate_loan_dataset(n=5000):
    age = np.random.randint(21, 65, n)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.6, 0.4])
    education = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.7, 0.3])
    employment = np.random.choice(['Salaried', 'Self-Employed', 'Business'], n, p=[0.5, 0.3, 0.2])
    income = np.random.randint(25000, 200000, n)
    co_applicant_income = np.random.randint(0, 80000, n)
    loan_amount = np.random.randint(50000, 1000000, n)
    loan_term = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240, 360], n)
    credit_score = np.random.randint(300, 900, n)
    existing_loans = np.random.randint(0, 5, n)
    property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.4, 0.35, 0.25])
    dependents = np.random.randint(0, 5, n)

    # Approval logic (realistic, not random)
    score = (
        (credit_score > 700).astype(int) * 3 +
        (income > 60000).astype(int) * 2 +
        (education == 'Graduate').astype(int) * 1 +
        (existing_loans < 2).astype(int) * 2 +
        (loan_amount < income * 5).astype(int) * 2 +
        (co_applicant_income > 0).astype(int) * 1 +
        (employment == 'Salaried').astype(int) * 1
    )

    noise = np.random.normal(0, 1, n)
    approval_prob = 1 / (1 + np.exp(-(score - 6 + noise)))
    approved = (approval_prob > 0.5).astype(int)

    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'education': education,
        'employment_type': employment,
        'income': income,
        'co_applicant_income': co_applicant_income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_score': credit_score,
        'existing_loans': existing_loans,
        'property_area': property_area,
        'dependents': dependents,
        'approved': approved
    })

    return df

if __name__ == "__main__":
    df = generate_loan_dataset(5000)
    df.to_csv("data/raw/loan_data.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df['approved'].value_counts())
    print(df.head())