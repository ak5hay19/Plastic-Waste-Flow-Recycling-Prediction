# Plastic Waste Flow & Recycling Prediction

**Advanced Data Analytics (CSE-AIML) UE23AM343AB1**
**Course Project - Aug-Dec 2025 (5th Semester)**

## Team Members
| Name | Section | SRN |
|------|---------|-----|
| TARUN S | A | PES1UG23AM919 |
| ADITYAA KUMAR H | A | PES1UG23AM025 |
| AKSHAY P SHETTI | A | PES1UG23AM039 |

## Project Overview

This project addresses the critical global issue of plastic waste mismanagement through comprehensive data analytics, predictive modeling, and interactive visualization. We analyze plastic waste generation, trade flows, and recycling patterns to provide actionable insights for policymakers and environmental organizations.

## Key Features

### 1. **Flow Analysis**
- Models plastic waste as a supply chain network
- Identifies inefficiencies in collection, processing, and recycling
- Graph-based analytics using NetworkX

### 2. **Predictive Analytics**
- Forecasts future plastic waste flows and recycling rates
- Implements multiple forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Prophet (Facebook's time series forecasting)
  - LSTM (Long Short-Term Memory neural networks)

### 3. **Hotspot Detection**
- Identifies geographic regions at highest risk
- Detects areas vulnerable to plastic leakage into oceans and landfills
- Clustering analysis for pattern recognition

### 4. **Anomaly Detection**
- Identifies unusual waste trade flows
- Detects potential illegal dumping or irregular patterns
- Statistical and ML-based detection (Isolation Forest)

### 5. **Interactive Dashboard**
- Consolidates all generated plots into a single, user-friendly HTML dashboard (`main_dashboard.html`).
- Allows for interactive exploration of maps, graphs, and forecasts.

## How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/ak5hay19/Plastic-Waste-Flow-Recycling-Prediction.git](https://github.com/ak5hay19/Plastic-Waste-Flow-Recycling-Prediction.git)
    cd Plastic-Waste-Flow-Recycling-Prediction
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main analysis script**
    ```bash
    python app.py
    ```

4.  **View the Dashboard**
    After the script finishes, it will print a clickable link to the main dashboard. You can also open it manually at:
    `outputs/figures/main_dashboard.html`

## Policy Recommendations

1. **Increase Recycling Infrastructure**: Countries with high mismanaged waste need immediate investment in recycling facilities
2. **Trade Regulation**: Implement stricter controls on international plastic waste trade to prevent illegal dumping
3. **Producer Responsibility**: Extended Producer Responsibility (EPR) schemes should be enforced globally
4. **Regional Cooperation**: Island nations and developing countries need international support and technology transfer
5. **Data-Driven Monitoring**: Establish real-time monitoring systems for plastic waste flows

## Future Work

- Real-time data integration from environmental sensors
- AI-powered waste segregation optimization
- Blockchain for transparent waste tracking
- Mobile app for citizen science data collection

## References

1. UN Comtrade Database: https://comtradeplus.un.org/
2. OECD Global Plastics Outlook: https://stats.oecd.org/
3. Our World in Data - Plastic Pollution: https://ourworldindata.org/plastic-pollution
4. Geyer, R., et al. (2017). "Production, use, and fate of all plastics ever made." Science Advances.

## License

This project is created for educational purposes as part of the Advanced Data Analytics course at PES University.

## Contact

For questions or collaborations, contact the team members through the university email system.

---
