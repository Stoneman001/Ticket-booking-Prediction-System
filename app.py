from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('ticket_demand_model.pkl')


@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    day_of_week = request.args.get('day', type=int)  # 1: Monday, 7: Sunday
    event_type = request.args.get('event', type=int)  # 1-5 Event Types

    # Create a DataFrame with zeros for all day_of_week and event_type columns
    feature_names = [f'day_of_week_{i}' for i in range(1, 8)] + [f'event_type_{i}' for i in range(1, 6)]
    input_features = pd.DataFrame(0, index=[0], columns=feature_names)

    # Set the correct columns for day_of_week and event_type
    if 1 <= day_of_week <= 7:
        input_features[f'day_of_week_{day_of_week}'] = 1
    if 1 <= event_type <= 5:
        input_features[f'event_type_{event_type}'] = 1

    # Predict demand
    predicted_demand = model.predict(input_features)[0]
    predicted_demand_rounded = round(predicted_demand)
    return jsonify({'predicted_demand': predicted_demand_rounded})


# Sample data
tickets = [
    {"id": 1, "event": "Concert", "available": 10},
    {"id": 2, "event": "Theater", "available": 5},
    {"id": 3, "event": "Bowling", "available": 4},
    {"id": 4, "event": "Car Ride", "available": 3},
    {"id": 5, "event": "Speed Train", "available": 2}

]

@app.route('/')
def index():
    return render_template('index.html', tickets=tickets)


@app.route('/book/<int:ticket_id>', methods=['POST'])
def book_ticket(ticket_id):
    for ticket in tickets:
        if ticket['id'] == ticket_id and ticket['available'] > 0:
            ticket['available'] -= 1
            return jsonify({"success": True, "ticket": ticket})
    return jsonify({"success": False, "message": "Ticket not available"})


if __name__ == '__main__':
    app.run(debug=True)


