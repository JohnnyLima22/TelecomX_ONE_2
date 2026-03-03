# business_simulation.py

def revenue_risk(probabilities, average_ticket):

    high_risk = probabilities > 0.65
    estimated_loss = high_risk.sum() * average_ticket * 12

    return estimated_loss