import csv

csvPath = "/Users/kourosh/Codes/Jupyter/AI/CA1/snacks.csv"

data = []
Weights = []
Values = []

with open(csvPath, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        data.append({'Snack': row['Snack'], 'Weight': int(row['Available Weight']), 'Value': int(row['Value']), 'Ratio': int(row['Value']) / int(row['Available Weight'])})

data_sorted = sorted(data, key=lambda x: x['Ratio'], reverse=True)

def fractional_knapsack_with_n_items(data, W, N):
    total_value = 0
    total_weight = 0
    items_selected = []

    for item in data:
        if total_weight < W and len(items_selected) < N:
            if total_weight + item['Weight'] <= W:
                total_value += item['Value']
                total_weight += item['Weight']
                items_selected.append((item['Snack'], item['Weight'], item['Value'], 1))
            else:
                fraction = (W - total_weight) / item['Weight']
                total_value += item['Value'] * fraction
                total_weight += item['Weight'] * fraction
                items_selected.append((item['Snack'], item['Weight'] * fraction, item['Value'] * fraction, fraction))
                break

    return total_value, total_weight, items_selected
