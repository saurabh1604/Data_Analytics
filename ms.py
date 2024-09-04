#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing essential libraries for data manipulation, visualization, and modeling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# For statistical tests and machine learning models
from sklearn.linear_model import LinearRegression  # For regression models
from sklearn.datasets import make_classification  # For classification dataset generation
from sklearn.cluster import KMeans  # For clustering models
# Import statsmodels for statistical analysis



# App Configurations
st.set_page_config(page_title="Data-Driven Decision Making", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Case Studies", "Automotive Industry", 
                                  "Real-Time Analytics", "Statistical Learning", 
                                  "Risk Management", "Collaborative Case Study"])

# Introduction Page
if page == "Introduction":
    st.title("Why Data-Driven Decision Making Matters")
    st.subheader("Explore the Power of Data Across Industries")

    # Evolution of Data-Driven Decision Making
    st.markdown("### The Evolution of Data-Driven Decision Making")
    st.markdown("""
        From intuitive decision-making to sophisticated data-driven strategies, industries across the world have transformed.
        Let's explore how data analytics has evolved in various sectors:
    """)

    # Interactive Storyboard for Evolution
    evolution = st.selectbox("Select an industry to learn how data transformed it:", 
                             ["Introduction", "Sports", "Finance", "Healthcare"])

    if evolution == "Introduction":
        st.write("""
        Data-driven decision-making has revolutionized industries globally. With more data than ever before, companies have shifted from relying on gut feeling to making strategic decisions backed by data.
        """)
    elif evolution == "Sports":
        st.write("""
        In sports, data analytics has enabled teams like Liverpool FC to refine their player recruitment, optimize match strategies, and track player performance. 
        Strategies like Moneyball in baseball have shown that data can lead to extraordinary success on and off the field.
        """)
        st.video('https://www.youtube.com/watch?v=CWnlGBVaRpQ')  # Example video
    elif evolution == "Finance":
        st.write("""
        Financial institutions leverage data analytics to predict stock trends, assess risk, and optimize portfolios. Firms like JP Morgan use data-driven algorithms to gain a competitive edge.
        """)
    elif evolution == "Healthcare":
        st.write("""
        In healthcare, data is used to predict disease outbreaks, personalize treatments, and improve patient outcomes. Hospitals such as John Hopkins employ predictive analytics to save lives.
        """)

    # Global Impact with Interactive Map
    st.markdown("### Global Impact of Data Analytics")
    st.write("Explore companies across the globe that lead the data revolution:")

    # Data for the map
    data = {
        "Country": ["USA", "UK", "Germany", "India", "China"],
        "Company": ["Google", "Tesco", "BMW", "TCS", "Alibaba"],
        "Industry": ["Tech", "Retail", "Automobile", "IT Services", "E-Commerce"],
        "Impact": ["Predictive advertising models", "Supply chain optimization", 
                   "Autonomous driving", "Big Data Analytics", "Customer behavior prediction"]
    }

    df = pd.DataFrame(data)

    # Interactive global map
    fig = px.scatter_geo(df, locations="Country", locationmode="country names", 
                         hover_name="Company", hover_data=["Industry", "Impact"], size_max=40)

    st.plotly_chart(fig)

    # Visualization of Data Growth Over Time
    st.markdown("### Exponential Growth of Data")
    st.write("See how the growth of data is reshaping industries:")

    years = np.arange(2000, 2025)
    data_volume = np.exp((years - 2000)/5)
    data_growth_df = pd.DataFrame({"Year": years, "Data Volume (Exabytes)": data_volume})

    fig_growth = px.line(data_growth_df, x="Year", y="Data Volume (Exabytes)", 
                         title="Exponential Growth of Data Over Time")
    st.plotly_chart(fig_growth)

    # Interactive Quiz
    st.markdown("### How Much Do You Know About Data Impact?")
    quiz_question = st.radio("Which industry has seen the highest impact from data analytics?", 
                             ["Healthcare", "Automobile", "Finance", "E-Commerce"])

    if quiz_question == "E-Commerce":
        st.success("Correct! E-Commerce giants like Amazon have used data to enhance customer experience, logistics, and predict trends.")
    else:
        st.error("Not quite. E-Commerce has been among the biggest beneficiaries of data analytics.")

# Case Studies Page (Placeholder)
elif page == "Case Studies":
    st.title("Case Studies: Data-Driven Success in the Automotive Sector")
    st.markdown("""
        Data analytics has revolutionized every facet of the automotive industry. Explore real-world case studies to see how data is transforming various business functions at leading companies like Maruti Suzuki, Toyota, and Tesla.
    """)

    # Select a Case Study Category
    case_study_area = st.selectbox("Choose a department to explore:", 
                                   ["Manufacturing", "HR", "Customer Experience", 
                                    "Supply Chain", "Sales & Marketing"])

    # Manufacturing Case Study (Automotive Focus)
    if case_study_area == "Manufacturing":
        st.subheader("Manufacturing: Optimizing Production with Data at Maruti Suzuki")
        st.write("""
            Data analytics is used to optimize production lines, reduce downtime, and improve quality control. Maruti Suzuki employs predictive maintenance, real-time machine monitoring, and IoT sensors to enhance productivity.
        """)

        # Interactive Chart for Production Efficiency at Maruti Suzuki
        st.markdown("### Production Efficiency Simulation")
        st.write("Adjust the production speed and quality control thresholds to see their impact on efficiency.")

        production_speed = st.slider("Production Speed (units/hour)", min_value=100, max_value=1000, step=50, value=600)
        quality_threshold = st.slider("Quality Control Threshold (%)", min_value=80, max_value=100, step=1, value=95)
        
        efficiency = production_speed * (quality_threshold / 100)
        st.write(f"Predicted Production Efficiency: **{efficiency:.2f} units/hour**")

        efficiency_df = pd.DataFrame({
            "Metric": ["Production Speed", "Quality Control", "Efficiency"],
            "Value": [production_speed, quality_threshold, efficiency]
        })
        fig = px.bar(efficiency_df, x="Metric", y="Value", title="Manufacturing Efficiency Optimization at Maruti Suzuki")
        st.plotly_chart(fig)

    # HR Case Study (Automotive Focus)
    elif case_study_area == "HR":
        st.subheader("HR: Reducing Employee Turnover at Toyota with Data Analytics")
        st.write("""
            Toyota uses predictive analytics to track employee satisfaction and engagement levels, identifying employees at risk of leaving. They implemented a data-driven retention strategy that reduced turnover by 15%.
        """)

        # Interactive Employee Turnover Simulation
        st.markdown("### Employee Turnover Prediction")
        st.write("Adjust employee satisfaction levels and predicted retention rates to simulate the impact of HR interventions.")

        satisfaction_level = st.slider("Employee Satisfaction (%)", min_value=0, max_value=100, step=5, value=75)
        turnover_rate = 100 - (satisfaction_level * 0.75)  # Simple model for turnover rate
        
        turnover_df = pd.DataFrame({
            "Metric": ["Satisfaction Level", "Predicted Turnover Rate"],
            "Value": [satisfaction_level, turnover_rate]
        })
        fig = px.bar(turnover_df, x="Metric", y="Value", title="Predicted Turnover Rate at Toyota")
        st.plotly_chart(fig)

    # Customer Experience Case Study (Automotive Focus)
    elif case_study_area == "Customer Experience":
        st.subheader("Customer Experience: Enhancing CX with Data at Tesla")
        st.write("""
            Tesla uses advanced analytics to predict customer preferences and personalize interactions. From personalized email campaigns to predicting when a customer is likely to buy a new vehicle, data helps Tesla maintain a high level of customer satisfaction.
        """)

        # Customer Satisfaction Prediction at Tesla
        st.markdown("### Predicting Customer Satisfaction Levels")
        st.write("Adjust customer interaction frequency and service quality to see how Tesla predicts customer satisfaction.")

        interaction_frequency = st.slider("Interaction Frequency (per month)", min_value=1, max_value=20, step=1, value=5)
        service_quality = st.slider("Service Quality Score (%)", min_value=70, max_value=100, step=1, value=90)
        
        satisfaction_score = (interaction_frequency * 5) + (service_quality * 0.5)
        st.write(f"Predicted Customer Satisfaction Score: **{satisfaction_score:.2f}**")

        satisfaction_df = pd.DataFrame({
            "Metric": ["Interaction Frequency", "Service Quality", "Satisfaction Score"],
            "Value": [interaction_frequency, service_quality, satisfaction_score]
        })
        fig = go.Figure(data=[
            go.Bar(name='Interaction Frequency', x=["Interaction Frequency"], y=[interaction_frequency], marker_color='lightskyblue'),
            go.Bar(name='Service Quality', x=["Service Quality"], y=[service_quality], marker_color='lightgreen'),
            go.Bar(name='Satisfaction Score', x=["Satisfaction Score"], y=[satisfaction_score], marker_color='lightcoral')
        ])
        fig.update_layout(title="Customer Satisfaction Prediction at Tesla", barmode='group')
        st.plotly_chart(fig)

    # Supply Chain Case Study (Automotive Focus)
    elif case_study_area == "Supply Chain":
        st.subheader("Supply Chain: Reducing Costs with Data at BMW")
        st.write("""
            BMW leverages data to optimize its supply chain, predicting demand and reducing stockouts. Data analytics also help streamline logistics, reduce costs, and ensure timely delivery of parts and vehicles.
        """)

        # Supply Chain Simulation for BMW
        st.markdown("### Optimizing Inventory Levels")
        st.write("Adjust the predicted demand and current stock to see the risk of stockouts in BMW’s supply chain.")

        stock_level = st.slider("Current Stock Level (units)", min_value=1000, max_value=50000, step=1000, value=20000)
        predicted_demand = st.slider("Predicted Demand (units)", min_value=1000, max_value=50000, step=1000, value=30000)
        stockout_risk = max(0, predicted_demand - stock_level)

        st.write(f"Stockout Risk: **{stockout_risk} units**")

        supply_chain_df = pd.DataFrame({
            "Scenario": ["Stock Level", "Predicted Demand", "Stockout Risk"],
            "Units": [stock_level, predicted_demand, stockout_risk]
        })
        fig = px.bar(supply_chain_df, x="Scenario", y="Units", title="Supply Chain Optimization for BMW")
        st.plotly_chart(fig)

    # Sales & Marketing Case Study (Automotive Focus)
    elif case_study_area == "Sales & Marketing":
        st.subheader("Sales & Marketing: Data-Driven Campaigns at Audi")
        st.write("""
            Audi uses customer behavior analytics and predictive models to optimize its marketing campaigns, focusing on high-value customers. Data analytics help Audi target the right customers with the right message, resulting in a 25% increase in sales from personalized campaigns.
        """)

        # Sales and Marketing Campaign Simulation
        st.markdown("### Simulating Sales Growth from Marketing Campaigns")
        st.write("Adjust marketing budget and campaign length to predict the increase in sales.")

        marketing_budget = st.slider("Marketing Budget ($)", min_value=10000, max_value=500000, step=5000, value=100000)
        campaign_length = st.slider("Campaign Length (weeks)", min_value=1, max_value=52, step=1, value=12)
        predicted_sales_growth = marketing_budget * 0.05 + campaign_length * 1000  # Simple formula for sales growth

        st.write(f"Predicted Sales Growth: **${predicted_sales_growth:,.2f}**")

        sales_growth_df = pd.DataFrame({
            "Metric": ["Marketing Budget", "Campaign Length", "Predicted Sales Growth"],
            "Value": [marketing_budget, campaign_length, predicted_sales_growth]
        })
        fig = px.bar(sales_growth_df, x="Metric", y="Value", title="Sales & Marketing Campaign Performance for Audi")
        st.plotly_chart(fig)

# Automotive Industry Page (Placeholder)
elif page == "Automotive Industry":
    st.title("Data Analytics in the Automotive Industry")
    st.markdown("""
        The automotive industry has embraced data analytics to make more informed and real-time decisions. 
        From optimizing production lines to improving customer satisfaction, data plays a crucial role in ensuring efficiency and competitiveness.
    """)

    # Key Data Elements in the Automotive Industry
    st.subheader("Key Data Elements Driving Decision-Making")
    st.write("""
        Data is collected from a variety of sources across the automotive industry. Here are the key data elements that are used to make decisions in different departments:
        
        - **Manufacturing Data**: Sensor data, machine performance, production rates, and quality control metrics.
        - **Sales & Marketing Data**: Customer demographics, buying patterns, marketing campaign performance, and sales growth trends.
        - **Supply Chain Data**: Inventory levels, logistics data, demand forecasting, and supplier performance.
        - **HR Data**: Employee satisfaction scores, retention rates, performance metrics, and hiring trends.
    """)

    # Real-Time Analytics
    st.subheader("The Role of Real-Time Analytics in Automotive Decision-Making")
    st.write("""
        Real-time analytics allows automotive companies to make on-the-spot decisions, optimizing performance and reducing inefficiencies. 
        It can be applied in various departments to achieve a variety of goals:
    """)

    # Examples of how real-time analytics is helping the automotive industry
    st.markdown("""
    **1. Manufacturing**: Real-time monitoring of machine performance helps predict failures, avoid downtime, and optimize production efficiency. 
    **2. Sales**: Real-time customer behavior tracking can be used to adjust marketing strategies dynamically, increasing the effectiveness of campaigns.
    **3. Supply Chain**: Tracking inventory in real-time allows companies to manage stock levels more efficiently, reducing costs associated with overstocking or stockouts.
    **4. HR**: Real-time tracking of employee performance and satisfaction helps companies make quick adjustments, improving retention and productivity.
    """)

    # Interactive Diagram of Real-Time Data Flow in the Automotive Industry
    st.subheader("Real-Time Data Flow in the Automotive Industry")
    st.write("Here’s how data flows through various departments in real-time, helping companies like Maruti Suzuki make data-driven decisions.")

    data_flow = {
        "Department": ["Manufacturing", "Sales & Marketing", "Supply Chain", "HR"],
        "Key Data": ["Sensor Data, Machine Performance", "Customer Data, Sales Trends", "Inventory Levels, Logistics", "Employee Performance, Satisfaction"],
        "Decision Impact": [
            "Optimizes production and reduces downtime",
            "Improves customer targeting and sales",
            "Enhances inventory management and reduces costs",
            "Improves employee retention and satisfaction"
        ]
    }

    data_flow_df = pd.DataFrame(data_flow)
    fig_flow = px.sunburst(data_flow_df, path=["Department", "Key Data", "Decision Impact"], title="Real-Time Data Flow in the Automotive Industry")
    st.plotly_chart(fig_flow)

    # Data Sources Across Departments
    st.subheader("Key Data Sources in the Automotive Industry")
    st.write("""
        Data is collected from a variety of sources across departments in an automotive company. These sources provide the raw data that drives effective decision-making.
    """)

    # Table of Data Sources
    data_sources = {
        "Department": ["Manufacturing", "Sales & Marketing", "Supply Chain", "HR"],
        "Data Sources": ["IoT Sensors, Machine Logs, QC Systems", "CRM Systems, Social Media, Customer Databases", 
                         "Inventory Systems, ERP, Supplier Data", "Employee Surveys, Performance Systems"]
    }
    data_sources_df = pd.DataFrame(data_sources)
    st.table(data_sources_df)

    # Visualizing the Decision-Making Process
    st.subheader("How Data is Used for Decision-Making")
    st.write("""
        Data from different departments is fed into analytics systems to help make more informed and accurate decisions. Let’s see how data is utilized across different areas of an automotive company:
    """)

    # Updated Data for Decision-Making Process Visualization
    decision_making = {
        "Stage": ["Data Collection", "Data Analysis", "Decision Implementation"],
        "Description": [
            "Data is collected from sensors, sales, inventory systems, and employee feedback.",
            "Data is analyzed using real-time analytics to generate insights and predictive models.",
            "Insights are implemented to optimize processes, reduce costs, and improve overall performance."
        ],
        "Impact": [25, 50, 75]  # Dummy impact percentages for visualization
    }

    decision_df = pd.DataFrame(decision_making)

    # Use a horizontal bar chart to display the decision-making process
    fig_decision = px.bar(decision_df, x="Impact", y="Stage", orientation='h', 
                          text="Description", title="Data-Driven Decision-Making Process",
                          labels={"Impact": "Decision-Making Impact (%)", "Stage": "Process Stage"})

    fig_decision.update_traces(textposition='outside')
    st.plotly_chart(fig_decision)

# Real-Time Analytics Page (Placeholder)
elif page == "Real-Time Analytics":
    st.title("The Power of Real-Time Analytics in Decision-Making")
    st.markdown("""
        Real-time analytics enables businesses to respond to data as it is generated, allowing for quick, informed decisions that improve efficiency, reduce downtime, and increase customer satisfaction.
    """)

    # Use Cases of Real-Time Analytics in Various Industries
    st.subheader("How Real-Time Data is Used Across Industries")
    st.write("""
        **1. Automotive Industry**: Real-time data from IoT sensors in manufacturing plants helps companies like Maruti Suzuki and Tesla monitor machine performance, detect faults early, and avoid costly downtimes.
        **2. Healthcare**: Hospitals like Mayo Clinic use real-time patient data to monitor vital signs, enabling doctors to provide immediate interventions and improve patient outcomes.
        **3. Logistics & Supply Chain**: Companies like FedEx and Amazon use real-time tracking data to optimize delivery routes, ensuring faster delivery times and minimizing fuel consumption.
        **4. Retail & E-Commerce**: Retailers leverage real-time customer data to adjust pricing dynamically, personalize customer experiences, and manage inventory more efficiently.
    """)

    # Explanation of how real-time analytics transforms industries
    st.markdown("""
    The benefits of real-time analytics can be seen across all sectors:
    
    - **Increased Efficiency**: By monitoring production lines in real time, companies can quickly identify and resolve issues before they cause delays.
    - **Cost Reduction**: Predictive maintenance powered by real-time data reduces the need for expensive repairs and downtime.
    - **Improved Customer Satisfaction**: Real-time data allows businesses to respond instantly to customer behavior, leading to better service and loyalty.
    """)

    # Interactive 3D Plotly Graph: Real-Time Data Simulation
    st.subheader("Real-Time Data Simulation: Impact on Automotive Production Efficiency")
    st.write("""
        Adjust production speed and machine health in this simulation to see how real-time analytics can optimize efficiency and reduce downtime.
    """)

    # Sliders for input
    production_speed = st.slider("Production Speed (units/hour)", min_value=100, max_value=1000, step=50, value=600)
    machine_health = st.slider("Machine Health (% of optimal performance)", min_value=50, max_value=100, step=5, value=90)

    # Calculate the efficiency based on the input
    efficiency = production_speed * (machine_health / 100)
    st.write(f"Predicted Production Efficiency: **{efficiency:.2f} units/hour**")

    # Generate some dummy real-time data for 3D visualization
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    z = np.outer(x, y) * (machine_health / 100) * (production_speed / 1000)

    fig_3d = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

    # Customize the layout of the 3D plot
    fig_3d.update_layout(
        title="3D Simulation of Production Efficiency with Real-Time Data",
        scene=dict(
            xaxis_title="Production Speed",
            yaxis_title="Machine Health",
            zaxis_title="Efficiency (units/hour)"
        ),
        width=700,
        height=700
    )

    # Render the 3D plot
    st.plotly_chart(fig_3d)

    # Additional Explanation of Real-Time Data Usage
    st.write("""
        This simulation shows how production efficiency is influenced by real-time data from both production speed and machine health. By monitoring these data points in real time, automotive manufacturers can optimize their processes, reduce downtime, and ensure consistent quality.
        
        Real-time analytics isn't limited to manufacturing. It's applied across industries to provide up-to-the-minute insights, enabling businesses to make data-driven decisions quickly and effectively. This ensures they remain competitive and efficient in a fast-paced world.
    """)

    # Final thoughts on the future of real-time analytics
    st.markdown("""
    ### The Future of Real-Time Analytics
    As data becomes more accessible and affordable to collect, real-time analytics will continue to revolutionize industries. The integration of AI and machine learning with real-time data will allow companies to automate decision-making, leading to even more significant gains in efficiency and profitability.
    """)


# Statistical Learning Page (Placeholder)
elif page == "Statistical Learning":
    st.title("Statistical Learning: From Basics to Advanced Models")
    st.markdown("""
        Statistical learning is the backbone of data-driven decision-making. In this section, we will explore how basic statistics, hypothesis testing, and advanced models such as regression, classification, and clustering can help drive insights in various scenarios.
    """)

    # Summary Statistics Section
    st.subheader("1. Summary Statistics")
    st.write("""
        Basic statistics such as mean, median, variance, and standard deviation help us summarize the distribution of data. 
        Let's start by looking at an example of mileage data for cars produced at two plants: Manesar and Pune.
    """)

    # Simulated mileage data for two plants
    np.random.seed(42)
    mileage_manesar = np.random.normal(loc=22, scale=3, size=1000)  # Mean = 22, SD = 3
    mileage_pune = np.random.normal(loc=20, scale=4, size=1000)  # Mean = 20, SD = 4

    summary_data = pd.DataFrame({
        "Plant": ["Manesar"] * 1000 + ["Pune"] * 1000,
        "Mileage": np.concatenate([mileage_manesar, mileage_pune])
    })

    # Display summary statistics using numpy
    summary_stats = pd.DataFrame({
        "Plant": ["Manesar", "Pune"],
        "Mean": [np.mean(mileage_manesar), np.mean(mileage_pune)],
        "Median": [np.median(mileage_manesar), np.median(mileage_pune)],
        "Std Dev": [np.std(mileage_manesar), np.std(mileage_pune)],
        "Variance": [np.var(mileage_manesar), np.var(mileage_pune)]
    })
    st.table(summary_stats)

    # Interactive 3D Boxplot
    st.subheader("Interactive 3D Boxplot for Mileage Distribution")
    st.write("""
        The boxplot provides a visual representation of the spread and skewness of the data. The interquartile range (IQR) helps in understanding how the data is distributed.
    """)

    # Create 3D Boxplot using Plotly
    fig_boxplot = go.Figure()
    fig_boxplot.add_trace(go.Box(y=mileage_manesar, name="Manesar", boxmean=True))
    fig_boxplot.add_trace(go.Box(y=mileage_pune, name="Pune", boxmean=True))

    # Update layout for 3D effect
    fig_boxplot.update_layout(
        title="3D Boxplot of Car Mileage from Manesar and Pune",
        scene=dict(
            xaxis=dict(title="Plant"),
            yaxis=dict(title="Mileage"),
        ),
        width=700,
        height=500
    )
    st.plotly_chart(fig_boxplot)

    # Hypothesis Testing: t-Test for Mileage
    st.subheader("2. Hypothesis Testing: t-Test for Mileage Comparison")
    st.write("""
        Now, let's perform a t-test to check if there is a significant difference in the average mileage of cars produced at the two plants (Manesar and Pune).
    """)

    # Perform a t-test using scipy
    t_stat, p_val = stats.ttest_ind(mileage_manesar, mileage_pune)

    st.write(f"**t-Statistic:** {t_stat:.2f}")
    st.write(f"**p-Value:** {p_val:.5f}")

    if p_val < 0.05:
        st.success("The p-value is less than 0.05, suggesting a statistically significant difference in mileage between the two plants.")
    else:
        st.info("The p-value is greater than 0.05, suggesting no statistically significant difference in mileage between the two plants.")

    # Visualization of t-Test Result
    st.subheader("Visualization of Mileage Comparison")
    fig_histogram = px.histogram(summary_data, x="Mileage", color="Plant", nbins=30, marginal="rug", title="Mileage Distribution for Manesar and Pune Plants")
    st.plotly_chart(fig_histogram)

    # Moving to Advanced Models: Regression, Classification, and Clustering
    st.subheader("3. Advanced Statistical Models")

    # Regression
    st.write("""
        **Regression Models** are used to predict continuous outcomes, such as predicting car sales based on various features like price, advertising budget, and customer reviews.
    """)

    # Simulate a regression example
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Random data for independent variable (e.g., Advertising Budget)
    y = 2.5 * X + np.random.randn(100, 1) * 2  # Dependent variable (e.g., Car Sales)

    # Fit a simple linear regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)

    st.write(f"**Regression Coefficient (Slope):** {reg.coef_[0][0]:.2f}")
    st.write(f"**Intercept:** {reg.intercept_[0]:.2f}")

    # Plot regression line
    fig_regression = px.scatter(x=X.flatten(), y=y.flatten(), trendline="ols", title="Linear Regression: Predicting Car Sales from Advertising Budget")
    st.plotly_chart(fig_regression)

    # Classification
    st.write("""
        **Classification Models** are used to predict categorical outcomes, such as classifying whether a car will be considered high-performance or not based on features like engine power and weight.
    """)

    # Simulate classification example (Car performance classification based on horsepower and weight)
    from sklearn.datasets import make_classification
    X_class, y_class = make_classification(n_samples=500, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
    
    fig_classification = px.scatter(x=X_class[:, 0], y=X_class[:, 1], color=y_class, title="Classification: High-Performance vs Low-Performance Cars")
    st.plotly_chart(fig_classification)

    # Clustering
    st.write("""
        **Clustering Models** are used to group similar data points together, such as grouping customers into different segments based on their purchasing habits or cars into clusters based on features like price, engine size, and mileage.
    """)

    # Simulate clustering example (Car segmentation based on price and mileage)
    from sklearn.cluster import KMeans
    X_clust = np.random.rand(200, 2) * [30000, 40]  # Random data for car price and mileage
    kmeans = KMeans(n_clusters=3)
    y_clust = kmeans.fit_predict(X_clust)

    fig_clustering = px.scatter(x=X_clust[:, 0], y=X_clust[:, 1], color=y_clust, title="Clustering: Segmenting Cars by Price and Mileage")
    st.plotly_chart(fig_clustering)

    # Final Thought on Statistical Learning
    st.markdown("""
    ### Conclusion
    Statistical learning provides powerful tools to draw insights from data, whether through basic statistics, hypothesis testing, or advanced models like regression, classification, and clustering. 
    These models are crucial for making data-driven decisions in various industries, particularly in the automotive sector.
    """)
# Risk Management Page (Placeholder)
elif page == "Risk Management":
    st.title("Risk Management Using Data")
    st.write("Coming Soon...")

# Collaborative Case Study Page (Placeholder)
elif page == "Collaborative Case Study":
    st.title("Collaborative Case Study")
    st.write("Coming Soon...")

