# sample_data.csv

Training dataset used to build the regression model in Azure ML Studio. Contains 1,000 employee records with `Name`, `Age`, `Department`, and `Salary` columns.

[⬇ Download sample_data.csv](../assets/files/sample_data.csv){ .md-button .md-button--primary download="sample_data.csv" }

---

## Preview (first 10 rows)

```
Name,Age,Department,Salary
Quincy,58,Legal,64769
Xander,46,Operations,79604
Frank,44,R&D,88645
Hannah,25,HR,61634
Leo,42,IT,69256
Frank,63,Finance,100789
Wendy,37,Sales,53274
Sam,33,Legal,108161
Quincy,59,Operations,67037
```

| Column | Type | Description |
|---|---|---|
| `Name` | string | Employee name |
| `Age` | integer | Employee age |
| `Department` | string | Department name (HR, IT, Finance, Sales, R&D, Legal, Operations, Engineering, Customer Service) |
| `Salary` | integer | Annual salary in USD |
