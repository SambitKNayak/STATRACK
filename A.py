from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
import streamlit as st
import numpy as np
import numpy_financial as npf
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns




def pca_to_1d(data):
    pca = PCA(n_components=1)
    transformed_data = pca.fit_transform(data)
    return transformed_data.flatten()

# Set page title and favicon
st.set_page_config(page_title="Stats & Finance Calculator", page_icon=":bar_chart:")


# Title
st.title("Statistics & Finance Calculator")

# Sidebar
st.sidebar.title("Select Calculator")
calculator_choice = st.sidebar.selectbox("Choose a calculator", ("Descriptive Statistics","Inferential Statistics", "Probability", "Mathematics", "2D Geometry","Advanced Statistics","Finance","Sotware Development Calculations"))

if calculator_choice == "Descriptive Statistics":
    st.header("Descriptive Statistics")

    # Data entry
    data = st.text_area("Enter comma-separated data:", height=100)

    def descriptive_statistics(data, measure):
        if measure == "Mean":
            return np.mean(data)
        elif measure == "Median":
            return np.median(data)
        elif measure == "Mode":
            return stats.mode(data)
        elif measure == "Range":
            return np.max(data) - np.min(data)
        elif measure == "Standard Deviation":
            return np.std(data)
        elif measure == "Variance":
            return np.var(data)
        elif measure == "Interquartile Range":
            return np.percentile(data, 75) - np.percentile(data, 25)
        elif measure == "25th Percentile":
            return np.percentile(data, 25)
        elif measure == "50th Percentile":
            return np.percentile(data, 50)
        elif measure == "75th Percentile":
            return np.percentile(data, 75)
        elif measure == "10th Percentile":
            return np.percentile(data, 10)
        elif measure == "90th Percentile":
            return np.percentile(data, 90)
        elif measure == "Frequency Distribution":
            return stats.relfreq(data, numbins=10).frequency
        elif measure == "Histogram":
            return np.histogram(data, bins=10)
        elif measure == "Skewness":
            return stats.skew(data)
        elif measure == "Kurtosis":
            return stats.kurtosis(data)
    
    # Statistical measure selection
    measure = st.selectbox("Select a measure", ("Mean", "Median", "Mode", "Range", "Standard Deviation", "Variance",
                                             "Interquartile Range", "25th Percentile","50th Percentile","75th Percentile",
                                             "10th Percentile", "90th Percentile", "Frequency Distribution",
                                             "Histogram", "Skewness", "Kurtosis"))
    # Calculate button
    if st.button("Calculate"):
        
        data = [float(x.strip()) for x in data.split(",") if x.strip()]
        
        if data:
            result = descriptive_statistics(data, measure)
            st.write(f"{measure}: {result}")
        else:
            st.warning("Please enter valid data.")


elif calculator_choice == "Inferential Statistics":
    st.header("Inferential Statistics")

    # Statistical calculation selection
    stat_calculation = st.selectbox("Select a calculation", ("T-Test", "ANOVA", "Chi-Square Test", "Simple Linear Regression", "Multiple Linear Regression", "Pearson Correlation", "Spearman Correlation", "Confidence Intervals"))

    if stat_calculation == "T-Test":
        st.subheader("T-Test")

        # Data entry
        st.write("Enter data for two groups:")
        data_group1 = st.text_area("Enter data for group 1 (comma-separated):")
        data_group2 = st.text_area("Enter data for group 2 (comma-separated):")

        data_group1 = [float(x.strip()) for x in data_group1.split(",") if x.strip()]
        data_group2 = [float(x.strip()) for x in data_group2.split(",") if x.strip()]

        if data_group1 and data_group2:  # Check if both groups have valid data
            # Perform T-Test using stats module
            t_statistic, p_value = stats.ttest_ind(data_group1, data_group2)
            st.write(f"T-Test Statistic: {t_statistic}")
            st.write(f"P-Value: {p_value}")
        else:
            st.warning("Please enter valid data for both groups.")

    elif stat_calculation == "ANOVA":
        st.subheader("ANOVA")

        # Data entry
        st.write("Enter data for multiple groups:")
        data_groups = []
        num_groups = st.number_input("Number of groups:", min_value=2, step=1, value=2)

        for i in range(num_groups):
            data_input = st.text_area(f"Enter data for group {i+1} (comma-separated):")
            data = [float(x.strip()) for x in data_input.split(",") if x.strip()]
            data_groups.append(data)

        if all(len(group) > 1 for group in data_groups):  # Check if all groups have valid data
            # Perform ANOVA using stats module
            f_statistic, p_value = stats.f_oneway(*data_groups)
            st.write(f"ANOVA F-Statistic: {f_statistic}")
            st.write(f"P-Value: {p_value}")
        else:
            st.warning("Please enter valid data for all groups.")

    elif stat_calculation == "Chi-Square Test":
        st.subheader("Chi-Square Test")

        # Data entry
        st.write("Enter observed and expected frequencies:")
        observed_data = st.text_area("Enter observed frequencies (comma-separated):")
        expected_data = st.text_area("Enter expected frequencies (comma-separated):")

        observed = [float(x.strip()) for x in observed_data.split(",") if x.strip()]
        expected = [float(x.strip()) for x in expected_data.split(",") if x.strip()]

        if observed and expected and len(observed) == len(expected):  # Check if both data sets are valid and have the same length
            # Perform Chi-Square Test using stats module
            chi_statistic, p_value = stats.chisquare(observed, f_exp=expected)
            st.write(f"Chi-Square Statistic: {chi_statistic}")
            st.write(f"P-Value: {p_value}")
        else:
            st.warning("Please enter valid observed and expected frequencies with the same length.")



    elif stat_calculation == "Simple Linear Regression":
        st.subheader("Simple Linear Regression")

        # Data entry
        st.write("Enter data for independent and dependent variables:")
        data_x = st.text_area("Enter data for independent variable X (comma-separated):")
        data_y = st.text_area("Enter data for dependent variable Y (comma-separated):")

        data_x = [float(x.strip()) for x in data_x.split(",") if x.strip()]
        data_y = [float(y.strip()) for y in data_y.split(",") if y.strip()]

        if data_x and data_y and len(data_x) == len(data_y):  # Check if both variables have valid data and the same length
            # Perform Simple Linear Regression using stats module
            slope, intercept, r_value, p_value, std_err = stats.linregress(data_x, data_y)
            st.write(f"Slope: {slope}")
            st.write(f"Intercept: {intercept}")
            st.write(f"R-squared: {r_value**2}")
            st.write(f"P-Value: {p_value}")
            st.write(f"Standard Error: {std_err}")

            # Frame equation: y = mx + b
            st.write(f"Equation of the line: y = {slope} * x + {intercept}")
            
            

    elif stat_calculation == "Multiple Linear Regression":
        st.subheader("Multiple Linear Regression")

        # Data entry
        st.write("Enter data for independent variables (features) and dependent variable:")
        data_input = st.text_area("Enter data (comma-separated, each row represents an observation):")
        data = [[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_input.split("\n") if row.strip()]

        if data:  # Check if data is not empty
            # Split data into independent and dependent variables
            X = np.array([row[:-1] for row in data])
            y = np.array([row[-1] for row in data])

            # Add a constant to the model (the intercept)
            X = sm.add_constant(X)

            # Perform Multiple Linear Regression
            model = sm.OLS(y, X).fit()
            st.write(model.summary())

            # Frame equation: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
            equation = f"y = {model.params[0]}"
            for i, coef in enumerate(model.params[1:], start=1):
                equation += f" + {coef}*x{i}"
            st.write(f"Equation of the multiple regression model: {equation}")
        else:
            st.warning("Please enter valid data.")
        

    elif stat_calculation == "Pearson Correlation":
        st.subheader("Pearson Correlation")

        # Data entry
        st.write("Enter two sets of data to calculate the Pearson Correlation:")
        data_set1 = st.text_area("Enter data set 1 (comma-separated):")
        data_set2 = st.text_area("Enter data set 2 (comma-separated):")

        data_set1 = [float(x.strip()) for x in data_set1.split(",") if x.strip()]
        data_set2 = [float(x.strip()) for x in data_set2.split(",") if x.strip()]

        if data_set1 and data_set2:  # Check if both sets have valid data
            # Calculate Pearson Correlation
            pearson_corr, p_value = stats.pearsonr(data_set1, data_set2)
            st.write(f"Pearson Correlation Coefficient: {pearson_corr}")
            st.write(f"P-Value: {p_value}")
        else:
            st.warning("Please enter valid data for both sets.")

    elif stat_calculation == "Spearman Correlation":
        st.subheader("Spearman Correlation")

        # Data entry
        st.write("Enter two sets of data to calculate the Spearman Correlation:")
        data_set1 = st.text_area("Enter data set 1 (comma-separated):")
        data_set2 = st.text_area("Enter data set 2 (comma-separated):")

        data_set1 = [float(x.strip()) for x in data_set1.split(",") if x.strip()]
        data_set2 = [float(x.strip()) for x in data_set2.split(",") if x.strip()]

        if data_set1 and data_set2:  # Check if both sets have valid data
            # Calculate Spearman Correlation
            spearman_corr, p_value = stats.spearmanr(data_set1, data_set2)
            st.write(f"Spearman Correlation Coefficient: {spearman_corr}")
            st.write(f"P-Value: {p_value}")
        else:
            st.warning("Please enter valid data for both sets.")


    elif stat_calculation == "Confidence Intervals":
        st.subheader("Confidence Intervals")

        # Data entry
        st.write("Enter data to calculate the confidence interval:")
        data = st.text_area("Enter data (comma-separated):")
        confidence_level = st.number_input("Enter confidence level (e.g., 0.95 for 95%):", min_value=0.0, max_value=1.0, value=0.95)

        data = [float(x.strip()) for x in data.split(",") if x.strip()]

        if data:  # Check if data is valid
            try:
                # Calculate Confidence Interval manually
                n = len(data)
                mean = np.mean(data)
                std_err = stats.sem(data)
                t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
                margin_of_error = t_critical * std_err
                ci_lower = mean - margin_of_error
                ci_upper = mean + margin_of_error
                st.write(f"Confidence Interval: ({ci_lower}, {ci_upper})")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter valid data.")
    
    
            
elif calculator_choice == "Probability":
    st.header("Probability Calculator")

    # Distribution selection
    distribution = st.selectbox("Select a distribution", ("Normal Distribution", "Uniform Distribution", "Exponential Distribution"))

    if distribution == "Normal Distribution":
        st.subheader("Normal Distribution")

        # User inputs
        mu = st.number_input("Mean (μ):", value=0.0)
        sigma = st.number_input("Standard Deviation (σ):", value=1.0)
        x = st.number_input("Value (x):")

        # Calculate probability
        cdf = stats.norm.cdf(x, mu, sigma)
        pdf = stats.norm.pdf(x, mu, sigma)
        ppf = stats.norm.ppf(cdf, mu, sigma)

        st.write(f"Cumulative Distribution Function (P(X ≤ {x})) = {cdf}")
        st.write(f"Probability Density Function (f(X) at {x}) = {pdf}")
        st.write(f"Inverse Cumulative Distribution Function (X for P(X ≤ {x})) = {ppf}")

    elif distribution == "Uniform Distribution":
        st.subheader("Uniform Distribution")

        # User inputs
        a = st.number_input("Lower Bound (a):", value=0.0)
        b = st.number_input("Upper Bound (b):", value=1.0)
        x = st.number_input("Value (x):")

        # Calculate probability
        cdf = stats.uniform.cdf(x, a, b)
        pdf = stats.uniform.pdf(x, a, b)
        ppf = stats.uniform.ppf(cdf, a, b)

        st.write(f"Cumulative Distribution Function (P(X ≤ {x})) = {cdf}")
        st.write(f"Probability Density Function (f(X) at {x}) = {pdf}")
        st.write(f"Inverse Cumulative Distribution Function (X for P(X ≤ {x})) = {ppf}")

    elif distribution == "Exponential Distribution":
        st.subheader("Exponential Distribution")

        # User inputs
        scale = st.number_input("Scale (β):", value=1.0)
        x = st.number_input("Value (x):")

        # Calculate probability
        cdf = stats.expon.cdf(x, scale=scale)
        pdf = stats.expon.pdf(x, scale=scale)
        ppf = stats.expon.ppf(cdf, scale=scale)

        st.write(f"Cumulative Distribution Function (P(X ≤ {x})) = {cdf}")
        st.write(f"Probability Density Function (f(X) at {x}) = {pdf}")
        st.write(f"Inverse Cumulative Distribution Function (X for P(X ≤ {x})) = {ppf}")


elif calculator_choice == "Mathematics":
    st.header("Mathematics Calculator")

    # Operation selection
    operation = st.selectbox("Select an operation", ("Addition", "Subtraction", "Multiplication", "Division", "Logarithm", "Exponentiation", "Linear Equation", "Quadratic Equation", "Polynomial Equation"))

    if operation == "Addition":
        num1 = st.number_input("Enter first number:")
        num2 = st.number_input("Enter second number:")
        if st.button("Calculate"):
            result = num1 + num2
            st.write("Result:", result)

    elif operation == "Subtraction":
        num1 = st.number_input("Enter first number:")
        num2 = st.number_input("Enter second number:")
        if st.button("Calculate"):
            result = num1 - num2
            st.write("Result:", result)

    elif operation == "Multiplication":
        num1 = st.number_input("Enter first number:")
        num2 = st.number_input("Enter second number:")
        if st.button("Calculate"):
            result = num1 * num2
            st.write("Result:", result)

    elif operation == "Division":
        num1 = st.number_input("Enter numerator:")
        num2 = st.number_input("Enter denominator:")
        if st.button("Calculate"):
            if num2 != 0:
                result = num1 / num2
                st.write("Result:", result)
            else:
                st.warning("Division by zero is not allowed.")
    
    elif operation == "Logarithm":
        num = st.number_input("Enter number:")
        base = st.number_input("Enter base:")
        if st.button("Calculate"):
            if num > 0 and base > 0 and base != 1:
                result = np.log(num) / np.log(base)
                st.write(f"Result: log_{base}({num}) =", result)
            else:
                st.warning("Invalid input. Ensure number and base are positive and base is not equal to 1.")

    elif operation == "Exponentiation":
        base = st.number_input("Enter base:")
        exponent = st.number_input("Enter exponent:")
        if st.button("Calculate"):
            result = base ** exponent
            st.write(f"Result: {base}^{exponent} =", result)

    elif operation == "Linear Equation":
        st.subheader("Solve Linear Equation (ax + b = 0)")
        a = st.number_input("Enter coefficient 'a':")
        b = st.number_input("Enter coefficient 'b':")
        if st.button("Calculate"):
            if a != 0:
                solution = -b / a
                st.write("Solution:", solution)
            else:
                st.warning("'a' cannot be zero. This is not a linear equation.")

    elif operation == "Quadratic Equation":
        st.subheader("Solve Quadratic Equation (ax^2 + bx + c = 0)")
        a = st.number_input("Enter coefficient 'a':")
        b = st.number_input("Enter coefficient 'b':")
        c = st.number_input("Enter coefficient 'c':")
        if st.button("Calculate"):
            discriminant = b**2 - 4*a*c
            if discriminant > 0:
                root1 = (-b + np.sqrt(discriminant)) / (2*a)
                root2 = (-b - np.sqrt(discriminant)) / (2*a)
                st.write("Root 1:", root1)
                st.write("Root 2:", root2)
            elif discriminant == 0:
                root = -b / (2*a)
                st.write("Double Root:", root)
            else:
                real_part = -b / (2*a)
                imag_part = np.sqrt(abs(discriminant)) / (2*a)
                st.write("Root 1:", real_part, "+", imag_part, "i")
                st.write("Root 2:", real_part, "-", imag_part, "i")

    elif operation == "Polynomial Equation":
        def solve_polynomial(coefficients):
            """
            Solves a polynomial equation with the given coefficients
            using companion matrices.
            """
            if not coefficients:
                return []
            
            degree = len(coefficients) - 1
            
            # If the leading coefficient is zero, return an empty list
            if coefficients[0] == 0:
                return []

            # Create the companion matrix
            companion_matrix = np.zeros((degree, degree))
            companion_matrix[0, :] = -np.array(coefficients[1:]) / coefficients[0]
            companion_matrix[np.arange(1, degree), np.arange(degree - 1)] = 1

            # Find the eigenvalues of the companion matrix
            roots = np.linalg.eigvals(companion_matrix)

            return roots
        st.subheader("Solve Polynomial Equation")
        degree = st.number_input("Enter the degree of the polynomial:", min_value=1, step=1)
        coefficients = []
        for i in range(degree + 1):
            coeff = st.number_input(f"Enter coefficient for x^{degree - i}:", key=f"coeff_{i}")
            coefficients.append(coeff)
        if st.button("Calculate"):
            roots = solve_polynomial(coefficients)
            st.write("Roots:", roots)

elif calculator_choice == "2D Geometry":
    st.header("2D Geometry Calculator")

    # Shape selection
    shape = st.selectbox("Select a shape", ("Line", "Circle", "Parabola", "Hyperbola"))

    if shape == "Line":
        st.subheader("Line")

        # User inputs
        slope = st.number_input("Slope (m):", value=1.0)
        intercept = st.number_input("Intercept (c):", value=0.0)

        # Plot line
        x = np.linspace(-10, 10, 100)
        y = slope * x + intercept
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Line: y = mx + c')
        st.pyplot()
        st.write("Equation: y = {:.2f}x + {:.2f}".format(slope, intercept))

    elif shape == "Circle":
        st.subheader("Circle")

        # User inputs
        radius = st.number_input("Radius (r):", value=1.0)
        center_x = st.number_input("Center X-coordinate (h):", value=0.0)
        center_y = st.number_input("Center Y-coordinate (k):", value=0.0)

        # Plot circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Circle: (x - h)^2 + (y - k)^2 = r^2')
        st.pyplot()
        st.write("Equation: (x - {:.2f})^2 + (y - {:.2f})^2 = {:.2f}^2".format(center_x, center_y, radius))

    elif shape == "Parabola":
        st.subheader("Parabola")

        # User inputs
        a = st.number_input("Coefficient 'a':", value=1.0)
        b = st.number_input("Coefficient 'b':", value=0.0)
        c = st.number_input("Coefficient 'c':", value=0.0)

        # Plot parabola
        x = np.linspace(-10, 10, 100)
        y = a * x**2 + b * x + c
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Parabola: y = ax^2 + bx + c')
        st.pyplot()
        st.write("Equation: y = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))

    elif shape == "Hyperbola":
        st.subheader("Hyperbola")

        # User inputs
        a = st.number_input("Coefficient 'a':", value=1.0)
        b = st.number_input("Coefficient 'b':", value=1.0)
        h = st.number_input("Center X-coordinate (h):", value=0.0)
        k = st.number_input("Center Y-coordinate (k):", value=0.0)

        # Plot hyperbola
        x = np.linspace(-10, 10, 400)
        y_pos = np.sqrt((x-h)**2 * (b**2/a**2) + k**2)
        y_neg = -np.sqrt((x-h)**2 * (b**2/a**2) + k**2)
        plt.plot(x, y_pos, label="y = sqrt((x-h)^2 * (b^2/a^2) + k^2)")
        plt.plot(x, y_neg, label="y = -sqrt((x-h)^2 * (b^2/a^2) + k^2)")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Hyperbola: (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1')
        plt.legend()
        st.pyplot()
        st.write("Equation: (x - {:.2f})^2 / {:.2f}^2 - (y - {:.2f})^2 / {:.2f}^2 = 1".format(h, a, k, b))


# New section for Advanced Statistics
elif calculator_choice == "Advanced Statistics":
    st.header("Advanced Statistics")

    # Statistical calculation selection
    stat_calculation = st.selectbox("Select a calculation", ("Covariance", "Correlation", "Coefficient of Correlation", "PCA", "SVM Hyperplane","K-means Clustering","Hierarchical Clustering(agglomerative)", "Hierarchical Clustering(divisive)","Hierarchical Clustering(divisive)[MATRIX]"))

    if stat_calculation == "Covariance":
        st.subheader("Covariance")

        # Data entry
        st.write("Enter data for two variables:")
        data_x = st.text_area("Enter data for variable X (comma-separated):")
        data_y = st.text_area("Enter data for variable Y (comma-separated):")

        data_x = [float(x.strip()) for x in data_x.split(",") if x.strip()]
        data_y = [float(y.strip()) for y in data_y.split(",") if y.strip()]

        if data_x and data_y:  # Check if both data sets are not empty
            # Calculate covariance
            covariance = np.cov(data_x, data_y)[0][1]
            st.write(f"Covariance: {covariance}")
        else:
            st.warning("Please enter valid data for both variables.")

    elif stat_calculation == "Correlation":
        st.subheader("Correlation")

        # Data entry
        st.write("Enter data for two variables:")
        data_x = st.text_area("Enter data for variable X (comma-separated):")
        data_y = st.text_area("Enter data for variable Y (comma-separated):")

        data_x = [float(x.strip()) for x in data_x.split(",") if x.strip()]
        data_y = [float(y.strip()) for y in data_y.split(",") if y.strip()]

        if data_x and data_y:  # Check if both data sets are not empty
            # Calculate correlation
            correlation = np.corrcoef(data_x, data_y)[0][1]
            st.write(f"Correlation: {correlation}")
        else:
            st.warning("Please enter valid data for both variables.")

    elif stat_calculation == "Coefficient of Correlation":
        st.subheader("Coefficient of Correlation")

        # Data entry
        st.write("Enter data for two variables:")
        data_x = st.text_area("Enter data for variable X (comma-separated):")
        data_y = st.text_area("Enter data for variable Y (comma-separated):")

        data_x = [float(x.strip()) for x in data_x.split(",") if x.strip()]
        data_y = [float(y.strip()) for y in data_y.split(",") if y.strip()]

        if data_x and data_y:  # Check if both data sets are not empty
            # Calculate coefficient of correlation
            coeff_corr = stats.pearsonr(data_x, data_y)[0]
            st.write(f"Coefficient of Correlation: {coeff_corr}")
        else:
            st.warning("Please enter valid data for both variables.")
            
    # elif stat_calculation == "SVM Hyperplane":
        
    #     # Function to calculate SVM hyperplane equation
    #     def svm_hyperplane(data_x, data_y):
    #         model = SVC(kernel='linear')
    #         model.fit(data_x, data_y)
    #         support_vectors = model.support_vectors_
    #         coefficients = model.coef_.reshape(-1, 1)
    #         intercept = model.intercept_
    #         return support_vectors, coefficients, intercept

    #     # SVM hyperplane calculation block
    #     st.subheader("SVM Hyperplane")

    #     # Data entry
    #     st.write("Enter data for two variables:")
    #     data_x = st.text_area("Enter data for variable X (comma-separated):")
    #     data_y = st.text_area("Enter data for variable Y (comma-separated):")

    #     data_x = np.array([float(x.strip()) for x in data_x.split(",") if x.strip()])
    #     data_y = np.array([float(y.strip()) for y in data_y.split(",") if y.strip()])

    #     if data_x.size > 0 and data_y.size > 0:  # Check if both data sets are not empty
    #         # Calculate SVM hyperplane
    #         support_vectors, coefficients, intercept = svm_hyperplane(data_x.reshape(-1, 1), data_y)
            
    #         # Determine positive and negative sided points
    #         positive_points = []
    #         negative_points = []
    #         # Determine positive and negative sided points
    #         positive_points = []
    #         negative_points = []
            
    #         for x, y in zip(data_x, data_y):
    #             point = np.array([x, y])
    #             if np.dot(coefficients, point) + intercept >= 0:
    #                 positive_points.append((x, y))
    #             else:
    #                 negative_points.append((x, y))
            
    #         # Frame the equation of the hyperplane
    #         equation = "Hyperplane Equation: "
    #         for i, coeff in enumerate(coefficients):
    #             equation += f"{coeff} * X{i+1} + "
    #         equation += f"{intercept} = 0"

    #         st.write("Support Vectors:")
    #         st.write(support_vectors)
    #         st.write("Positive-sided Points:")
    #         st.write(positive_points)
    #         st.write("Negative-sided Points:")
    #         st.write(negative_points)
    #         st.write(equation)
    #     else:
    #         st.warning("Please enter valid data for both variables.")



    # Streamlit code for SVM hyperplane calculation block
    elif stat_calculation == "SVM Hyperplane":
        
        # Function to calculate SVM hyperplane equation
        def svm_hyperplane(data_x, data_y):
            model = SVC(kernel='linear')
            model2 = SVR(kernel='linear')
            model.fit(data_x, data_y)
            model2.fit(data_x, data_y)
            support_vectors = model.support_vectors_
            coefficients = model.coef_[0]  # Get the first (and only) row of coefficients
            coefficients1 = model.coef_  # Get all row of coefficients
            intercept = model.intercept_[0]  # Get the scalar value of intercept
            intercept1 = model.intercept_  # Get the scalar value of intercept
            # -----------------------------------
            support_vectors2 = model2.support_vectors_
            coefficients2 = model2.coef_[0]  # Get the first (and only) row of coefficients
            intercept2 = model2.intercept_  # Get the scalar value of intercept
            return support_vectors, coefficients, coefficients1, intercept, intercept1, support_vectors2, coefficients2, intercept2
        
        st.subheader("SVM Hyperplane")

        st.write("""
            The code I wrote here pertains to linear SVM (Support Vector Machine) with a linear kernel using both SVClassification & SVRegression. Linear SVM seeks to find a hyperplane that best separates the classes in the feature space. The hyperplane equation is determined by the coefficients (weights) and the intercept, which are optimized during the training process to maximize the margin between the classes.

            Here's a breakdown of the components:

            1. **Support Vector Machine (SVM)**: This is the general machine learning algorithm used for classification tasks. SVM aims to find the optimal hyperplane that separates classes in the feature space.

            2. **Linear Kernel**: In the provided code, we used a linear kernel (`kernel='linear'`) when initializing the SVM model. A linear kernel implies that the decision boundary is a hyperplane in the original feature space.

            3. **SVM Hyperplane Equation**: The function `svm_hyperplane` calculates the coefficients (weights) and the intercept of the hyperplane. These parameters are essential for defining the hyperplane equation:

            w_1x_1 + w_2x_2 + ... + w_nx_n + b = 0

            Here, w_1, w_2, ..., w_n are the coefficients (weights) obtained from the SVM model, x_1, x_2, ..., x_n are the features, and b is the intercept.

            Overall, the provided code specifically deals with linear SVM and the calculation of the hyperplane equation for linearly separable data.""")
        
        # Data entry
        st.write("Enter data for two variables:")
        data_x = st.text_area("Enter data for variable X (comma-separated):")
        data_y = st.text_area("Enter data for variable Y (comma-separated):")

        # Convert the input data to numpy arrays
        data_x = np.array([float(x.strip()) for x in data_x.split(",") if x.strip()])
        data_y = np.array([float(y.strip()) for y in data_y.split(",") if y.strip()])

        if data_x.size > 0 and data_y.size > 0:  # Check if both data sets are not empty
            # Calculate SVM hyperplane
            support_vectors, coefficients, coefficients1, intercept, intercept1, support_vectors2, coefficients2, intercept2 = svm_hyperplane(data_x.reshape(-1, 1), data_y)

            # Construct and display the hyperplane equations:
            
            #for SVC USED:
            hyperplane_equation = f"y = {coefficients[0]}x + {intercept}"
            st.write("The hyperplane equation ( using SVC ) is:")
            st.write("GENERAL  --Writing only first coefficient & intercept-- ")
            st.latex(hyperplane_equation)
            
            hyperplane_equation0 = f"y = {coefficients[0]}x + {intercept1}"
            st.write("The hyperplane equation ( using SVC ) is:")
            st.write("Writing only first coefficient & all intercepts-- ")
            st.latex(hyperplane_equation0)
            
            hyperplane_equation1 = f"y = {coefficients1}x + {intercept1}"
            st.write("Writing all coefficients and intercepts-- ")
            st.latex(hyperplane_equation1)
            
            st.write("Support Vectors:", support_vectors)
            st.write("Coefficients:", coefficients1)
            st.write("Intercept:", intercept1)
            
            # # Display the results
            # st.write("SVC used:")
            # st.write("Support Vectors:", support_vectors)
            # st.write("Coefficients:", coefficients)
            # st.write("Intercept:", intercept)
            
            #for SVR USED:
            hyperplane_equation2 = f"y = {coefficients2}x + {intercept2}"
            st.write("The hyperplane equation ( using SVR ) is:")
            st.latex(hyperplane_equation2)
            
            # Display the results
            st.write("SVR used:")
            st.write("Support Vectors:", support_vectors2)
            st.write("Coefficients:", coefficients2)
            st.write("Intercept:", intercept2)
        else:
            st.warning("Please enter valid data for both variables.")
            
            
    # elif stat_calculation == "PCA":
        # def pca_to_1d(data):
        #     pca = PCA(n_components=1)
        #     transformed_data = pca.fit_transform(data)
        #     return transformed_data.flatten()
    #     st.subheader('Method 1: Enter data points in brackets')
    #     st.write('Enter data points in the format: [x1, y1], [x2, y2], ..., [xn, yn]')
    #     data_input1 = st.text_input('Enter data points:')
        
    #     st.subheader('Method 2: Enter data in textarea')
    #     st.write("Enter data:")
    #     data_input2 = st.text_area("Enter data (comma-separated)[row->sample, column->feature]:\n"+"eg: 2 features 1 sample: 4,11 ")
    #     data_input2 = [[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_input2.split("\n") if row.strip()]
        
    #     if st.button('Calculate PCA (Method 1)'):
    #         try:
    #             # Parse input string to extract data points
    #             data_points = eval(data_input1)
    #             data = np.array(data_points)

    #             # Perform PCA for dimensionality reduction from 2 to 1
    #             transformed_data = pca_to_1d(data)
                
    #             st.write('Transformed Data (1D):')
    #             st.write(transformed_data)
    #         except Exception as e:
    #             st.error(f'An error occurred: {e}')
        
    #     if st.button('Calculate PCA (Method 2)'):
    #         try:
    #             if data_input2:  # Check if data is not empty
    #                 # Perform PCA
    #                 pca = PCA(n_components=2)
    #                 pca.fit(data_input2)
    #                 transformed_data = pca.transform(data_input2)

    #                 # Display results
    #                 st.write("Transformed Data:")
    #                 st.write(pd.DataFrame(transformed_data, columns=['PC1', 'PC2']))
    #             else:
    #                 st.warning("Please enter valid data.")
    #         except Exception as e:
    #             st.error(f'An error occurred: {e}')
    
    # ------------------------------------------------------------------------
    elif stat_calculation == "PCA":
        
        st.subheader("PCA (Principal Component Analysis)")

        # User input for the number of features and desired components
        num_features = st.number_input("Enter the number of features you have:", min_value=1, step=1)
        num_components = st.number_input("Enter the number of components you want after PCA:", min_value=1, step=1)

        # Data entry
        st.write("Enter data:")
        data_input = st.text_area("Enter data (comma-separated)[row->sample, column->feature]:\n" + "eg: 3 features 1 sample: 4,11,3 ")
        data_input = [[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_input.split("\n") if row.strip()]

        if st.button("Calculate PCA"):
            try:
                if data_input:  # Check if data is not empty
                    # Perform PCA
                    pca = PCA(n_components=num_components)
                    transformed_data = pca.fit_transform(data_input)

                    # Display results
                    st.write(f"Transformed Data ({num_components}D):")
                    st.write(pd.DataFrame(transformed_data, columns=[f'PC{i}' for i in range(1, num_components + 1)]))
                else:
                    st.warning("Please enter valid data.")
            except Exception as e:
                st.error(f'An error occurred: {e}')
                
                
                
    elif stat_calculation == "K-means Clustering":
        
        # Function to perform K-means clustering
        def k_means_clustering(data, num_clusters, initial_centroids):
            kmeans = KMeans(n_clusters=num_clusters, init=initial_centroids, n_init=1)
            kmeans.fit(data)
            clusters = kmeans.predict(data)
            centroids = kmeans.cluster_centers_
            return clusters, centroids
        
        st.subheader("K-means Clustering")

        # Data entry
        st.write("Enter data points (comma-separated):")
        data_input = st.text_area("Enter data points (one point per line):")
        data = np.array([[float(val) for val in row.split(",")] for row in data_input.split("\n") if row.strip()])

        num_clusters = st.number_input("Enter the number of clusters:", min_value=1, step=1)

        st.write("Enter initial centroids:")
        initial_centroids = []
        for i in range(num_clusters):
            centroid_input = st.text_input(f"Enter initial centroid for cluster {i+1} (comma-separated):")
            if centroid_input:
                centroid = [float(val) for val in centroid_input.split(",")]
                initial_centroids.append(centroid)

        initial_centroids = np.array(initial_centroids)

        if st.button("Perform K-means Clustering"):
            try:
                if data.size > 0 and initial_centroids.size > 0:  # Check if data and initial centroids are not empty
                    clusters, centroids = k_means_clustering(data, num_clusters, initial_centroids)
                    st.write("Clustered Points:")
                    for i in range(num_clusters):
                        st.write(f"Cluster {i+1}:")
                        cluster_points = data[clusters == i]
                        st.write(cluster_points)
                    st.write("Centroids:")
                    st.write(centroids)
                else:
                    st.warning("Please enter valid data and initial centroids.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
                
# --------------------------------------------------------  
    # elif stat_calculation == "Hierarchical Clustering":

    #     def hierarchical_clustering_1d(data_points_1d, linkage_method):
    #         # Reshape the data points to a 2D array
    #         data_points_1d = data_points_1d.reshape(-1, 1)

    #         # Calculate the pairwise Euclidean distance
    #         distances_1d = pdist(data_points_1d)

    #         # Perform hierarchical clustering using the linkage method
    #         linkage_matrix_1d = linkage(distances_1d, method=linkage_method)

    #         # Plot the dendrogram
    #         plt.figure(figsize=(10, 5))
    #         dendrogram(linkage_matrix_1d, labels=[f'{point[0]:.2f}' for point in data_points_1d], leaf_rotation=90)
    #         plt.title(f'Hierarchical Clustering Dendrogram (1D) - Method: {linkage_method}')
    #         plt.xlabel('Data Points')
    #         plt.ylabel('Distance')
    #         plt.xticks(rotation=45)
    #         st.pyplot()

    #         # Determine clusters using a distance threshold or number of clusters
    #         threshold = st.number_input("Enter distance threshold for cluster formation:")
    #         clusters = fcluster(linkage_matrix_1d, threshold, criterion='distance')

    #         st.write("Clusters formed by points:")
    #         unique_clusters = np.unique(clusters)
    #         for cluster_id in unique_clusters:
    #             cluster_points = [point[0] for point, cluster in zip(data_points_1d, clusters) if cluster == cluster_id]
    #             st.write(f"Cluster {cluster_id}: {cluster_points}")

    #     def hierarchical_clustering_2d(data_points_2d, linkage_method):
    #         # Calculate the pairwise Euclidean distance
    #         distances = pdist(data_points_2d)

    #         # Perform hierarchical clustering using the linkage method
    #         linkage_matrix = linkage(distances, method=linkage_method)

    #         # Plot the dendrogram
    #         plt.figure(figsize=(10, 5))
    #         dendrogram(linkage_matrix, labels=[f'{point[0]:.2f}, {point[1]:.2f}' for point in data_points_2d], leaf_rotation=90)
    #         plt.title(f'Hierarchical Clustering Dendrogram (2D) - Method: {linkage_method}')
    #         plt.xlabel('Data Points')
    #         plt.ylabel('Distance')
    #         plt.xticks(rotation=45)
    #         st.pyplot()

    #         # Determine clusters using a distance threshold or number of clusters
    #         threshold = st.number_input("Enter distance threshold for cluster formation:")
    #         clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    #         st.write("Clusters formed by points:")
    #         unique_clusters = np.unique(clusters)
    #         for cluster_id in unique_clusters:
    #             cluster_points = [point for point, cluster in zip(data_points_2d, clusters) if cluster == cluster_id]
    #             st.write(f"Cluster {cluster_id}: {cluster_points}")

    #     # User interface for selecting data dimension
    #     data_dimension = st.radio("Select data dimension:", ("1D", "2D"))

    #     # Available linkage methods
    #     linkage_methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

    #     # User interface for selecting linkage method
    #     linkage_method = st.selectbox("Select linkage method:", linkage_methods)

    #     if data_dimension == "1D":
    #         st.write("### Hierarchical Clustering (1D)")
    #         st.write("Enter one-dimensional data points separated by commas (,):")
    #         data_points_1d = st.text_input("Data Points:")
    #         data_points_1d = [float(point.strip()) for point in data_points_1d.split(',') if point.strip()]

    #         if data_points_1d:
    #             hierarchical_clustering_1d(np.array(data_points_1d), linkage_method)
    #         else:
    #             st.warning("Please enter valid data points.")

    #     elif data_dimension == "2D":
    #         st.write("### Hierarchical Clustering (2D)")
    #         st.write("Enter two-dimensional data points separated by commas (,):")
    #         data_points_2d = st.text_area("Data Points:")
    #         data_points_2d = [[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_points_2d.split("\n") if row.strip()]

    #         if data_points_2d:
    #             hierarchical_clustering_2d(np.array(data_points_2d), linkage_method)
    #         else:
    #             st.warning("Please enter valid data points.")

# ---------------------------------------------------------------

    elif stat_calculation == "Hierarchical Clustering(agglomerative)":
        
        def hierarchical_clustering_1d(data_points_1d, linkage_method):
            # Reshape the data points to a 2D array
            data_points_1d = data_points_1d.reshape(-1, 1)

            # Calculate the pairwise Euclidean distance
            distances_1d = pdist(data_points_1d)

            # Perform hierarchical clustering using the linkage method
            linkage_matrix_1d = linkage(distances_1d, method=linkage_method)

            # Plot the dendrogram
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix_1d, labels=[f'{point[0]:.2f}' for point in data_points_1d], leaf_rotation=90, ax=ax)
            ax.set_title(f'Hierarchical Clustering Dendrogram (1D) - Method: {linkage_method}')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Distance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Determine clusters using a distance threshold or number of clusters
            threshold = st.number_input("Enter distance threshold for cluster formation:")
            clusters = fcluster(linkage_matrix_1d, threshold, criterion='distance')

            st.write("Clusters formed by points:")
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters:
                cluster_points = [point[0] for point, cluster in zip(data_points_1d, clusters) if cluster == cluster_id]
                st.write(f"Cluster {cluster_id}: {cluster_points}")

        def hierarchical_clustering_2d(data_points_2d, linkage_method):
            # Calculate the pairwise Euclidean distance
            distances = pdist(data_points_2d)

            # Perform hierarchical clustering using the linkage method
            linkage_matrix = linkage(distances, method=linkage_method)

            # Plot the dendrogram
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, labels=[f'{point[0]:.2f}, {point[1]:.2f}' for point in data_points_2d], leaf_rotation=90, ax=ax)
            ax.set_title(f'Hierarchical Clustering Dendrogram (2D) - Method: {linkage_method}')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Distance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Determine clusters using a distance threshold or number of clusters
            threshold = st.number_input("Enter distance threshold for cluster formation:")
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')

            st.write("Clusters formed by points:")
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters:
                cluster_points = [point for point, cluster in zip(data_points_2d, clusters) if cluster == cluster_id]
                st.write(f"Cluster {cluster_id}: {cluster_points}")


        # User interface
        st.title("Agglomerative Hierarchical Clustering")
        # User interface for selecting data dimension
        data_dimension = st.radio("Select data dimension:", ("1D", "2D"))

        # Available linkage methods
        linkage_methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

        # User interface for selecting linkage method
        linkage_method = st.selectbox("Select linkage method:", linkage_methods)

        if data_dimension == "1D":
            st.write("### Hierarchical Clustering (1D)")
            st.write("Enter one-dimensional data points separated by commas (,):")
            data_points_1d = st.text_input("Data Points:")
            data_points_1d = [float(point.strip()) for point in data_points_1d.split(',') if point.strip()]

            if data_points_1d:
                hierarchical_clustering_1d(np.array(data_points_1d), linkage_method)
            else:
                st.warning("Please enter valid data points.")

        elif data_dimension == "2D":
            st.write("### Hierarchical Clustering (2D)")
            st.write("Enter two-dimensional data points separated by commas (,):")
            data_points_2d = st.text_area("Data Points:")
            data_points_2d = [[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_points_2d.split("\n") if row.strip()]

            if data_points_2d:
                hierarchical_clustering_2d(np.array(data_points_2d), linkage_method)
            else:
                st.warning("Please enter valid data points.")

# -----------------------------------------------------------------

    # elif stat_calculation == "Hierarchical Clustering(divisive)":
        
    #     def divisive_hierarchical_clustering_1d(data_points_1d, threshold):
    #         # Reshape the data points to a 2D array
    #         data_points_1d = data_points_1d.reshape(-1, 1)

    #         # Start with all points in one cluster
    #         cluster_assignments = np.zeros(len(data_points_1d))

    #         # Plot the initial state
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.scatter(data_points_1d, np.zeros_like(data_points_1d), c=cluster_assignments, cmap='viridis')
    #         ax.set_title('Initial State (1D)')
    #         ax.set_xlabel('Data Points')
    #         ax.set_ylabel('Cluster')
    #         plt.xticks(rotation=45)
    #         st.pyplot(fig)

    #         # Perform divisive hierarchical clustering
    #         while len(np.unique(cluster_assignments)) > 1:
    #             # Calculate the pairwise Euclidean distance
    #             distances_1d = pdist(data_points_1d)

    #             # Calculate the linkage matrix
    #             linkage_matrix_1d = linkage(distances_1d, method='average')

    #             # Find the cluster with the highest average dissimilarity
    #             max_cluster_index = np.argmax(np.average(linkage_matrix_1d[:, 2].reshape(-1, 1), axis=1))

    #             # Split the cluster into two subclusters
    #             max_cluster_indices = np.where(cluster_assignments == max_cluster_index)[0]
    #             cluster_assignments[max_cluster_indices[:len(max_cluster_indices) // 2]] = max(cluster_assignments) + 1

    #             # Plot the current state
    #             fig, ax = plt.subplots(figsize=(10, 5))
    #             ax.scatter(data_points_1d, np.zeros_like(data_points_1d), c=cluster_assignments, cmap='viridis')
    #             ax.set_title(f'Divisive Hierarchical Clustering (1D) - Threshold: {threshold}')
    #             ax.set_xlabel('Data Points')
    #             ax.set_ylabel('Cluster')
    #             plt.xticks(rotation=45)
    #             st.pyplot(fig)

    #         # Output final clusters
    #         st.write("Final Clusters formed by points:")
    #         unique_clusters = np.unique(cluster_assignments)
    #         for cluster_id in unique_clusters:
    #             cluster_points = [point[0] for point, cluster in zip(data_points_1d, cluster_assignments) if cluster == cluster_id]
    #             st.write(f"Cluster {cluster_id}: {cluster_points}")

    #     def divisive_hierarchical_clustering_2d(data_points_2d, threshold):
    #         # Start with all points in one cluster
    #         cluster_assignments = np.zeros(len(data_points_2d))

    #         # Plot the initial state
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=cluster_assignments, cmap='viridis')
    #         ax.set_title('Initial State (2D)')
    #         ax.set_xlabel('Data Points')
    #         ax.set_ylabel('Cluster')
    #         plt.xticks(rotation=45)
    #         st.pyplot(fig)

    #         # Perform divisive hierarchical clustering
    #         while len(np.unique(cluster_assignments)) > 1:
    #             # Calculate the pairwise Euclidean distance
    #             distances = pdist(data_points_2d)

    #             # Calculate the linkage matrix
    #             linkage_matrix = linkage(distances, method='average')

    #             # Find the cluster with the highest average dissimilarity
    #             max_cluster_index = np.argmax(np.average(linkage_matrix[:, 2].reshape(-1, 1), axis=1))

    #             # Split the cluster into two subclusters
    #             max_cluster_indices = np.where(cluster_assignments == max_cluster_index)[0]
    #             cluster_assignments[max_cluster_indices[:len(max_cluster_indices) // 2]] = max(cluster_assignments) + 1

    #             # Plot the current state
    #             fig, ax = plt.subplots(figsize=(10, 5))
    #             ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=cluster_assignments, cmap='viridis')
    #             ax.set_title(f'Divisive Hierarchical Clustering (2D) - Threshold: {threshold}')
    #             ax.set_xlabel('Data Points')
    #             ax.set_ylabel('Cluster')
    #             plt.xticks(rotation=45)
    #             st.pyplot(fig)

    #         # Output final clusters
    #         st.write("Final Clusters formed by points:")
    #         unique_clusters = np.unique(cluster_assignments)
    #         for cluster_id in unique_clusters:
    #             cluster_points = [point for point, cluster in zip(data_points_2d, cluster_assignments) if cluster == cluster_id]
    #             st.write(f"Cluster {cluster_id}: {cluster_points}")

    #     # User interface
    #     st.title("Divisive Hierarchical Clustering (Divisive)")

    #     # Select data dimension
    #     data_dimension = st.radio("Select data dimension:", ("1D", "2D"))

    #     # Enter distance threshold for cluster formation
    #     threshold = st.number_input("Enter distance threshold for cluster formation:")

    #     if data_dimension == "1D":
    #         st.write("### Divisive Hierarchical Clustering (1D)")
    #         st.write("Enter one-dimensional data points separated by commas (,):")
    #         data_points_1d = st.text_input("Data Points:")
    #         data_points_1d = np.array([float(point.strip()) for point in data_points_1d.split(',') if point.strip()])

    #         if data_points_1d.size > 0:
    #             divisive_hierarchical_clustering_1d(data_points_1d, threshold)
    #         else:
    #             st.warning("Please enter valid data points.")

    #     elif data_dimension == "2D":
    #         st.write("### Divisive Hierarchical Clustering (2D)")
    #         st.write("Enter two-dimensional data points separated by commas (,):")
    #         data_points_2d = st.text_area("Data Points:")
    #         data_points_2d = np.array([[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_points_2d.split("\n") if row.strip()])

    #         if data_points_2d.size > 0:
    #             divisive_hierarchical_clustering_2d(data_points_2d, threshold)
    #         else:
    #             st.warning("Please enter valid data points.")
    # --------------------------------------------------
    
    elif stat_calculation == "Hierarchical Clustering(divisive)":

        def divisive_hierarchical_clustering(data, threshold, dimension):
            # Perform divisive hierarchical clustering
            linkage_matrix = linkage(data, method='average')

            # Plot the dendrogram
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, orientation='top', leaf_font_size=10,
                    labels=np.array([f"{point[0]}, {point[1]}" for point in data]))
            ax.set_title('Divisive Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Distance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Determine clusters based on the threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')

            # Output final clusters
            st.write("Final Clusters formed by points:")
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters:
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_points = [data[i] for i in cluster_indices]
                st.write(f"Cluster {cluster_id}: {cluster_points}")

        # User interface
        st.title("Divisive Hierarchical Clustering (Divisive)")

        # Select data dimension
        data_dimension = st.radio("Select data dimension:", ("1D", "2D"))

        # Enter distance threshold for cluster formation
        threshold = st.number_input("Enter distance threshold for cluster formation:")

        if data_dimension == "1D":
            st.write("### Divisive Hierarchical Clustering (1D)")
            st.write("Enter one-dimensional data points separated by commas (,):")
            data_points_1d = st.text_input("Data Points:")
            data_points_1d = np.array([float(point.strip()) for point in data_points_1d.split(',') if point.strip()])

            if data_points_1d.size > 0:
                divisive_hierarchical_clustering(data_points_1d.reshape(-1, 1), threshold, dimension="1D")
            else:
                st.warning("Please enter valid data points.")

        elif data_dimension == "2D":
            st.write("### Divisive Hierarchical Clustering (2D)")
            st.write("Enter two-dimensional data points separated by commas (,):")
            data_points_2d = st.text_area("Data Points:")
            data_points_2d = np.array([[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_points_2d.split("\n") if row.strip()])

            if data_points_2d.size > 0:
                divisive_hierarchical_clustering(data_points_2d, threshold, dimension="2D")
            else:
                st.warning("Please enter valid data points.")

# --------------------------------------------------------------------------------------- 
    # elif stat_calculation == "Hierarchical Clustering(divisive)":   
        # def divisive_hierarchical_clustering_1d(data_points_1d, threshold):
        #     # Start with all points in one cluster
        #     cluster_assignments = np.zeros(len(data_points_1d))

        #     # Plot the initial state
        #     fig, ax = plt.subplots(figsize=(10, 5))
        #     dendrogram(linkage(data_points_1d.reshape(-1, 1), method='average'), ax=ax, color_threshold=threshold)
        #     ax.set_title('Initial State (1D)')
        #     ax.set_xlabel('Data Points')
        #     ax.set_ylabel('Cluster')
        #     plt.xticks(rotation=45)
        #     st.pyplot(fig)

        #     # Perform divisive hierarchical clustering
        #     while len(np.unique(cluster_assignments)) > 1:
        #         # Calculate the pairwise Euclidean distance
        #         distances_1d = np.diff(sorted(data_points_1d[cluster_assignments == 0]))

        #         # Calculate the linkage matrix
        #         linkage_matrix_1d = linkage(distances_1d, method='average')

        #         # Find the cluster with the highest average dissimilarity
        #         max_cluster_index = np.argmax(np.average(linkage_matrix_1d[:, 2].reshape(-1, 1), axis=1))

        #         # Split the cluster into two subclusters
        #         max_cluster_indices = np.where(cluster_assignments == max_cluster_index)[0]
        #         cluster_assignments[max_cluster_indices[:len(max_cluster_indices) // 2]] = max(cluster_assignments) + 1

        #         # Plot the current state
        #         fig, ax = plt.subplots(figsize=(10, 5))
        #         dendrogram(linkage_matrix_1d, ax=ax, color_threshold=threshold)
        #         ax.set_title(f'Divisive Hierarchical Clustering (1D) - Threshold: {threshold}')
        #         ax.set_xlabel('Data Points')
        #         ax.set_ylabel('Cluster')
        #         plt.xticks(rotation=45)
        #         st.pyplot(fig)

        #     # Output final clusters
        #     st.write("Final Clusters formed by points:")
        #     unique_clusters = np.unique(cluster_assignments)
        #     for cluster_id in unique_clusters:
        #         cluster_points = [point[0] for point, cluster in zip(data_points_1d, cluster_assignments) if cluster == cluster_id]
        #         st.write(f"Cluster {cluster_id}: {cluster_points}")

        # def divisive_hierarchical_clustering_2d(data_points_2d, threshold):
        #     # Start with all points in one cluster
        #     cluster_assignments = np.zeros(len(data_points_2d))

        #     # Plot the initial state
        #     fig, ax = plt.subplots(figsize=(10, 5))
        #     dendrogram(linkage(data_points_2d, method='average'), ax=ax, color_threshold=threshold)
        #     ax.set_title('Initial State (2D)')
        #     ax.set_xlabel('Data Points')
        #     ax.set_ylabel('Cluster')
        #     plt.xticks(rotation=45)
        #     st.pyplot(fig)

        #     # Perform divisive hierarchical clustering
        #     while len(np.unique(cluster_assignments)) > 1:
        #         # Calculate the pairwise Euclidean distance
        #         distances = np.linalg.norm(data_points_2d[cluster_assignments == 0, np.newaxis, :] - data_points_2d[np.newaxis, :, :], axis=-1)

        #         # Calculate the linkage matrix
        #         linkage_matrix = linkage(distances, method='average')

        #         # Find the cluster with the highest average dissimilarity
        #         max_cluster_index = np.argmax(np.average(linkage_matrix[:, 2].reshape(-1, 1), axis=1))

        #         # Split the cluster into two subclusters
        #         max_cluster_indices = np.where(cluster_assignments == max_cluster_index)[0]
        #         cluster_assignments[max_cluster_indices[:len(max_cluster_indices) // 2]] = max(cluster_assignments) + 1

        #         # Plot the current state
        #         fig, ax = plt.subplots(figsize=(10, 5))
        #         dendrogram(linkage_matrix, ax=ax, color_threshold=threshold)
        #         ax.set_title(f'Divisive Hierarchical Clustering (2D) - Threshold: {threshold}')
        #         ax.set_xlabel('Data Points')
        #         ax.set_ylabel('Cluster')
        #         plt.xticks(rotation=45)
        #         st.pyplot(fig)

        #     # Output final clusters
        #     st.write("Final Clusters formed by points:")
        #     unique_clusters = np.unique(cluster_assignments)
        #     for cluster_id in unique_clusters:
        #         cluster_points = [point for point, cluster in zip(data_points_2d, cluster_assignments) if cluster == cluster_id]
        #         st.write(f"Cluster {cluster_id}: {cluster_points}")

        # # User interface
        # st.title("Divisive Hierarchical Clustering (Divisive)")

        # # Select data dimension
        # data_dimension = st.radio("Select data dimension:", ("1D", "2D"))

        # # Enter distance threshold for cluster formation
        # threshold = st.number_input("Enter distance threshold for cluster formation:")

        # if data_dimension == "1D":
        #     st.write("### Divisive Hierarchical Clustering (1D)")
        #     st.write("Enter one-dimensional data points separated by commas (,):")
        #     data_points_1d = st.text_input("Data Points:")
        #     data_points_1d = np.array([float(point.strip()) for point in data_points_1d.split(',') if point.strip()])

        #     if data_points_1d.size > 0:
        #         divisive_hierarchical_clustering_1d(data_points_1d, threshold)
        #     else:
        #         st.warning("Please enter valid data points.")

        # elif data_dimension == "2D":
        #     st.write("### Divisive Hierarchical Clustering (2D)")
        #     st.write("Enter two-dimensional data points separated by commas (,):")
        #     data_points_2d = st.text_area("Data Points:")
        #     data_points_2d = np.array([[float(val.strip()) for val in row.split(",") if val.strip()] for row in data_points_2d.split("\n") if row.strip()])

        #     if data_points_2d.size > 0:
        #         divisive_hierarchical_clustering_2d(data_points_2d, threshold)
        #     else:
        #         st.warning("Please enter valid data points.")

    
    # ----------------------------------------------------------------------------
   
    elif stat_calculation == "Hierarchical Clustering(divisive)[MATRIX]":
        def divisive_hierarchical_clustering_distance_matrix(dist_matrix, variables, threshold):
            # Perform divisive hierarchical clustering
            linkage_matrix = linkage(dist_matrix, method='average')

            # Plot the dendrogram
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, labels=variables, leaf_rotation=90, ax=ax)
            plt.title('Divisive Hierarchical Clustering Dendrogram')
            plt.xlabel('Data Points')
            plt.ylabel('Distance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Determine clusters based on the threshold
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')

            # st.write("Clusters formed by points:")
            # unique_clusters = np.unique(clusters)
            # for cluster_id in unique_clusters:
            #     cluster_points = [variable for variable, cluster in zip(variables, clusters) if cluster == cluster_id]
            #     st.write(f"Cluster {cluster_id}: {cluster_points}")


            # Extract clustered variables for each cluster
            clustered_variables = {}
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                clustered_vars = [variables[i] for i in cluster_indices]
                clustered_variables[cluster_id] = clustered_vars

            st.write("Clusters formed by points:")
            for cluster_id, vars_in_cluster in clustered_variables.items():
                st.write(f"Cluster {cluster_id}: {vars_in_cluster}")
            
        # User interface
        st.title("Divisive Hierarchical Clustering with Distance Matrix")

        # Enter distance threshold for cluster formation
        threshold = st.number_input("Enter distance threshold for cluster formation:")

        # Option to provide distance matrix
        if st.checkbox("Provide distance matrix"):
            num_variables = st.number_input("Enter the number of variables:", min_value=1, value=2, step=1)

            # Input distance matrix
            dist_matrix = np.zeros((num_variables, num_variables))
            for i in range(num_variables):
                for j in range(i+1, num_variables):
                    dist_matrix[i, j] = st.number_input(f"Distance between {i+1} and {j+1}:", value=1.0)

            # Fill in symmetric part of the distance matrix
            dist_matrix = dist_matrix + dist_matrix.T

            # Input variable names
            variables = []
            for i in range(num_variables):
                variables.append(st.text_input(f"Variable {i+1} name:", f"Variable {i+1}"))

            if st.button("Cluster"):
                divisive_hierarchical_clustering_distance_matrix(dist_matrix, variables, threshold)

    # -----------------------------------------------------------------------------

elif calculator_choice == "Finance":

    st.header("Finance Calculator")

    # Financial calculation selection
    finance_calculation = st.selectbox("Select a calculation", ("Percentage","Interests(&rates) Calculator","Loan Calculator","Investment Calculator","Loan Payment Schedule","Profit/Loss","Payback Period", "Net Present Value (NPV)", "Internal Rate of Return (IRR)"))

    if finance_calculation == "Percentage":
        st.header("Percentage Calculator")
        
        fin = st.selectbox("Select a percentage calculation",("Simple","Expenditure Reduction","Consumption Increase", 
                                                           "Population Growth", "Depreciation",
                                                           "Successive Percentage Increase",
                                                           "Successive Percentage Decrease",
                                                           "Net Change - Successive Increase then Decrease",
                                                           "Net Change - Successive Decrease then Increase",
                                                           "Equivalent Decrease - Successive Increase then Decrease"))

        def percentage_to_fraction(percentage):
            return percentage / 100

        def fraction_to_percentage(fraction):
            return fraction * 100

        def expenditure_reduction_due_to_price_increase(price_increase_percentage):
            return (price_increase_percentage / (100 + price_increase_percentage)) * 100

        def consumption_increase_due_to_price_decrease(price_decrease_percentage):
            return (price_decrease_percentage / (100 - price_decrease_percentage)) * 100

        def population_after_years(initial_population, annual_growth_rate, years):
            return initial_population * ((1 + (annual_growth_rate / 100)) ** years)

        def population_before_years(final_population, annual_growth_rate, years):
            return final_population / ((1 + (annual_growth_rate / 100)) ** years)

        def value_after_depreciation(initial_value, annual_depreciation_rate, years):
            return initial_value * ((1 - (annual_depreciation_rate / 100)) ** years)

        def value_before_depreciation(final_value, annual_depreciation_rate, years):
            return final_value / ((1 - (annual_depreciation_rate / 100)) ** years)

        def successive_percentage_increase(a_percentage, b_percentage):
            return a_percentage + b_percentage + ((a_percentage * b_percentage) / 100)

        def successive_percentage_decrease(a_percentage, b_percentage):
            return a_percentage + b_percentage - ((a_percentage * b_percentage) / 100)

        def net_change_successive_increase_decrease(a_percentage, b_percentage):
            return a_percentage - b_percentage - ((a_percentage * b_percentage) / 100)

        def net_change_successive_decrease_increase(a_percentage, b_percentage):
            return b_percentage - a_percentage - ((a_percentage * b_percentage) / 100)

        def equivalent_decrease_after_successive_increase_decrease(percentage):
            return (percentage / 10) ** 2
        
        if fin == "Simple":
            part = st.number_input("Enter the amount or part of the total:",value = 0.0)
            total = st.number_input("Enter the total amount:",value = 0.0)
            if part and total:
                percentage = (part / total) * 100
                st.write(f"Percentage: {percentage}")
            
        elif fin == "Expenditure Reduction":
            price_increase_percentage = st.number_input("Enter price increase percentage:", value=0.0)
            expenditure_reduction = expenditure_reduction_due_to_price_increase(price_increase_percentage)
            st.write("Necessary reduction in consumption to avoid increase in expenditure:", expenditure_reduction, "%")

        elif fin == "Consumption Increase":
            price_decrease_percentage = st.number_input("Enter price decrease percentage:", value=0.0)
            consumption_increase = consumption_increase_due_to_price_decrease(price_decrease_percentage)
            st.write("Necessary increase in consumption to keep the same expenditure:", consumption_increase, "%")

        elif fin == "Population Growth":
            initial_population = st.number_input("Enter initial population:", value=0)
            annual_growth_rate = st.number_input("Enter annual growth rate percentage:", value=0.0)
            years = st.number_input("Enter number of years:", value=0)
            population_after = population_after_years(initial_population, annual_growth_rate, years)
            population_before = population_before_years(population_after, annual_growth_rate, years)
            st.write("Population after", years, "years:", population_after)
            st.write("Population before", years, "years:", population_before)

        elif fin == "Depreciation":
            initial_value = st.number_input("Enter initial value:", value=0.0)
            annual_depreciation_rate = st.number_input("Enter annual depreciation rate percentage:", value=0.0)
            years = st.number_input("Enter number of years:", value=0)
            value_after = value_after_depreciation(initial_value, annual_depreciation_rate, years)
            value_before = value_before_depreciation(value_after, annual_depreciation_rate, years)
            st.write("Value after", years, "years:", value_after)
            st.write("Value before", years, "years:", value_before)

        elif fin == "Successive Percentage Increase":
            a_percentage = st.number_input("Enter first percentage increase:", value=0.0)
            b_percentage = st.number_input("Enter second percentage increase:", value=0.0)
            net_increase = successive_percentage_increase(a_percentage, b_percentage)
            st.write("Net increase after successive percentage increase:", net_increase, "%")

        elif fin == "Successive Percentage Decrease":
            a_percentage = st.number_input("Enter first percentage decrease:", value=0.0)
            b_percentage = st.number_input("Enter second percentage decrease:", value=0.0)
            net_decrease = successive_percentage_decrease(a_percentage, b_percentage)
            st.write("Net decrease after successive percentage decrease:", net_decrease, "%")

        elif fin == "Net Change - Successive Increase then Decrease":
            a_percentage = st.number_input("Enter first percentage increase:", value=0.0)
            b_percentage = st.number_input("Enter second percentage decrease:", value=0.0)
            net_change = net_change_successive_increase_decrease(a_percentage, b_percentage)
            st.write("Net change after successive increase then decrease:", net_change, "%")

        elif fin == "Net Change - Successive Decrease then Increase":
            a_percentage = st.number_input("Enter first percentage decrease:", value=0.0)
            b_percentage = st.number_input("Enter second percentage increase:", value=0.0)
            net_change = net_change_successive_decrease_increase(a_percentage, b_percentage)
            st.write("Net change after successive decrease then increase:", net_change, "%")

        elif fin == "Equivalent Decrease - Successive Increase then Decrease":
            percentage = st.number_input("Enter percentage for successive increase then decrease:", value=0.0)
            equivalent_decrease = equivalent_decrease_after_successive_increase_decrease(percentage)
            st.write("Equivalent decrease after successive increase then decrease:", equivalent_decrease, "%")


# --------------------------------------------------------

    elif finance_calculation == "Profit/Loss":
        st.header("Profit/Loss Calculator")
        
        fin = st.selectbox("Select a calculation", ("Profit/Loss", "Profit/Loss Percentage",
                                                               "Selling Price", "Cost Price", "Selling Price (after discount)",
                                                               "Cost Price (after discount)", "Discount"))
        
        if fin == "Profit/Loss":
            def calculate_profit_loss(selling_price, cost_price):
                return selling_price - cost_price
            st.subheader("Profit/Loss Calculation")
            cost_price = st.number_input("Enter Cost Price:")
            selling_price = st.number_input("Enter Selling Price:")
            
            st.write("if value is +ve means profit")
            st.write("if value is -ve means loss")
            
            value = calculate_profit_loss(selling_price, cost_price)
            if value >= 0:
                st.write(f"Profit: {value}")
            else:
                st.write(f"Loss: {value}")
        
        elif fin == "Profit/Loss Percentage":
            def calculate_profit_loss(selling_price, cost_price):
                return selling_price - cost_price
            def calculate_profit_loss_percentage(value, cost_price):
                return (value / cost_price) * 100
            
            st.subheader("Profit/Loss Percentage Calculation")
            cost_price = st.number_input("Enter Cost Price:")
            selling_price = st.number_input("Enter Selling Price:")
            value = calculate_profit_loss(selling_price, cost_price)
            
            st.write("if percentage is +ve means profit percent")
            st.write("if percentage is -ve means loss percent")
            
            profit_loss_percentage = calculate_profit_loss_percentage(value, cost_price)
            st.write(f"Profit/Loss Percentage: {profit_loss_percentage}%")
            
        elif fin == "Selling Price":
            def calculate_selling_price(cost_price, profit_percentage):
                return ((100 + profit_percentage) / 100) * cost_price

            st.subheader("Selling Price Calculation")
            cost_price = st.number_input("Enter Cost Price:")
            profit_percentage = st.number_input("Enter Profit Percentage:")
            selling_price = calculate_selling_price(cost_price, profit_percentage)
            st.write(f"Selling Price: {selling_price}")
        
        elif fin == "Cost Price":
            def calculate_cost_price(selling_price, profit_percentage):
                return (100 / (100 + profit_percentage)) * selling_price
            
            st.subheader("Cost Price Calculation")
            selling_price = st.number_input("Enter Selling Price:")
            profit_percentage = st.number_input("Enter Profit Percentage:")
            cost_price = calculate_cost_price(selling_price, profit_percentage)
            st.write(f"Cost Price: {cost_price}")
            
        elif fin == "Selling Price (after discount)":
            def calculate_selling_price_discount(cost_price, loss_percentage):
                return ((100 - loss_percentage) / 100) * cost_price

            st.subheader("Selling Price (after discount) Calculation")
            cost_price = st.number_input("Enter Cost Price:")
            loss_percentage = st.number_input("Enter Loss Percentage:")
            selling_price_discount = calculate_selling_price_discount(cost_price, loss_percentage)
            st.write(f"Selling Price (after discount): {selling_price_discount}")
        
        elif finance_calculation == "Cost Price (after discount)":
            def calculate_cost_price_discount(selling_price, loss_percentage):
                return (100 / (100 - loss_percentage)) * selling_price
            
            st.subheader("Cost Price (after discount) Calculation")
            selling_price = st.number_input("Enter Selling Price:")
            loss_percentage = st.number_input("Enter Loss Percentage:")
            cost_price_discount = calculate_cost_price_discount(selling_price, loss_percentage)
            st.write(f"Cost Price (after discount): {cost_price_discount}")
        
        elif finance_calculation == "Discount":
            def calculate_discount(marked_price, selling_price):
                return marked_price - selling_price
            
            st.subheader("Discount Calculation")
            marked_price = st.number_input("Enter Marked Price:")
            selling_price = st.number_input("Enter Selling Price:")
            discount = calculate_discount(marked_price, selling_price)
            st.write(f"Discount: {discount}")
    # ------------------------------------------------
    
    elif finance_calculation == "Interests(&rates) Calculator":
        st.header("Interests Calculator")
        fin = st.selectbox("Select an interest calculation", ("Simple Interest", "Compound Interest", "Nominal Interest Rate", "Real Interest Rate", "Effective Interest Rate", "Fixed Interest Rate", "Variable Interest Rate", "Floating Interest Rate"))
        
        def rate_converter(period_from, rate_from, period_to):
            if period_from == period_to:
                return rate_from
            if period_from == "Yearly":
                if period_to == "Years":
                    return rate_from
                elif period_to == "Months":
                    return rate_from / 12
                elif period_to == "Weeks":
                    return rate_from / 52
                elif period_to == "Days":
                    return rate_from / 365
                elif period_to == "Quarters":
                    return rate_from / 4
            elif period_from == "Monthly":
                if period_to == "Months":
                    return rate_from
                elif period_to == "Years":
                    return rate_from * 12
                elif period_to == "Weeks":
                    return rate_from / 4
                elif period_to == "Days":
                    return rate_from / 30  # Approximation
                elif period_to == "Quarters":
                    return rate_from * 3
            elif period_from == "Weekly":
                if period_to == "Weeks":
                    return rate_from
                elif period_to == "Years":
                    return rate_from * 52
                elif period_to == "Months":
                    return rate_from * 4
                elif period_to == "Days":
                    return rate_from / 7
                elif period_to == "Quarters":
                    return rate_from * 13  # Approximation
            elif period_from == "Daily":
                if period_to == "Days":
                    return rate_from
                elif period_to == "Years":
                    return rate_from * 365
                elif period_to == "Months":
                    return rate_from * 30  # Approximation
                elif period_to == "Weeks":
                    return rate_from * 7
                elif period_to == "Quarters":
                    return rate_from * 91  # Approximation
            elif period_from == "Quarterly":
                if period_to == "Quarters":
                    return rate_from
                elif period_to == "Years":
                    return rate_from * 4
                elif period_to == "Months":
                    return rate_from / 3
                elif period_to == "Weeks":
                    return rate_from / 13  # Approximation
                elif period_to == "Days":
                    return rate_from / 91  # Approximation
            
            # Add handling for additional periods as needed
            
        if fin == "Simple Interest":
            st.subheader("Simple Interest Calculator")
            # Input fields for simple interest calculation
            principal_amount = st.number_input("Enter the principal amount")
            rate_period = st.selectbox("Select rate of interest", ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"])
            interest_rate = st.number_input("Enter the interest rate (%)")
            time_period = st.selectbox("Select Time Period", ["Years", "Quarters", "Months", "Weeks", "Days", "Hours"])
            no_of_time_period = st.number_input("Enter the time period")

            # Calculate button
            # if st.button("Calculate"):
                # interest_rate_now = rate_converter(rate_period, interest_rate, time_period)
                
                # # Perform simple interest calculation
                # interest = (principal_amount * interest_rate_now * no_of_time_period) / 100
                # total_amount = principal_amount + interest
                
                # st.write(f"After {no_of_time_period} {time_period} at {interest_rate}% interest/{rate_period}")
                # st.write("Interest : ", interest)
                # st.write("Amount : ", total_amount)
        
            # Calculate button
            if st.button("Calculate"):
                interest_rate_now = rate_converter(rate_period, interest_rate, time_period)
                
                # Lists to store data for visualization
                periods = []
                total_amounts = []
                interests = []

                for t in range(1, int(no_of_time_period) + 1):
                    periods.append(t)
                    interest = (principal_amount * interest_rate_now * t) / 100
                    total_amount = principal_amount + interest
                    interests.append(interest)
                    total_amounts.append(total_amount)
                    
                final_interest = interests[-1]
                final_amount = total_amounts[-1]
                st.write("Final Interest:", final_interest)
                st.write("Final Amount:", final_amount)
                    
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(periods, total_amounts, label="Total Amount")
                plt.plot(periods, interests, label="Interest")
                plt.xlabel("Time Period")
                plt.ylabel("Amount")
                plt.title("Interest and Amount over Time (Simple Interest)")
                plt.legend()
                st.pyplot(fig)

                # Create DataFrame for table
                data = {
                    "Time Period": periods,
                    "Total Amount": total_amounts,
                    "Interest": interests
                }
                df = pd.DataFrame(data)

                # Display table
                st.write(df)
        
        elif fin == "Compound Interest":
            st.subheader("Compound Interest Calculator")
            # Input fields for compound interest calculation
            principal_amount = st.number_input("Enter the principal amount")
            interest_rate_period = st.selectbox("Select rate of interest", ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"])
            interest_rate = st.number_input("Enter the interest rate (%)")
            time_period = st.selectbox("Select Time Period", ["Years", "Quarters", "Months", "Weeks", "Days", "Hours"])
            no_of_time_period = st.number_input("Enter the time period")

            # Calculate button
            # if st.button("Calculate"):
                # # Convert interest rate to the selected time period
                # interest_rate_now = rate_converter(interest_rate_period, interest_rate, time_period)
                
                # # Perform compound interest calculation
                # amount = principal_amount * (1 + (interest_rate_now / 100)) ** no_of_time_period
                # interest = amount - principal_amount
                
                # st.write(f"After {no_of_time_period} {time_period} at {interest_rate}% interest/{interest_rate_period}")
                # st.write("Total amount:", amount)
                # st.write("Interest:", interest)

            # Calculate button
            if st.button("Calculate"):
                interest_rate_now = rate_converter(interest_rate_period, interest_rate, time_period)
                
                # Lists to store data for visualization
                periods = []
                total_amounts = []
                interests = []

                for t in range(1, int(no_of_time_period) + 1):
                    periods.append(t)
                    total_amount = principal_amount * (1 + (interest_rate_now / 100)) ** t
                    interest = total_amount - principal_amount
                    interests.append(interest)
                    total_amounts.append(total_amount)
                    
                final_interest = interests[-1]
                final_amount = total_amounts[-1]
                st.write("Final Interest:", final_interest)
                st.write("Final Amount:", final_amount)
                
                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(periods, total_amounts, label="Total Amount")
                plt.plot(periods, interests, label="Interest")
                plt.xlabel("Time Period")
                plt.ylabel("Amount")
                plt.title("Interest and Amount over Time (Compound Interest)")
                plt.legend()
                st.pyplot(fig)

                # Create DataFrame for table
                data = {
                    "Time Period": periods,
                    "Total Amount": total_amounts,
                    "Interest": interests
                }
                df = pd.DataFrame(data)

                # Display table
                st.write(df)
    # --------------------------------------------------------
        elif fin == "Nominal Interest Rate":
            st.subheader("Nominal Interest Rate Converter")
            # Input fields for nominal interest rate calculation
            effective_interest_rate = st.number_input("Enter the effective interest rate (%)")
            inflation_rate = st.number_input("Enter the inflation rate (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform nominal interest rate calculation
                nominal_interest_rate = effective_interest_rate + inflation_rate
                st.write("Nominal Interest Rate:", nominal_interest_rate)
        
    
        elif fin == "Real Interest Rate":
            st.subheader("Real Interest Rate Calculator")
            # Input fields for real interest rate calculation
            nominal_interest_rate = st.number_input("Enter the nominal interest rate (%)")
            inflation_rate = st.number_input("Enter the inflation rate (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform real interest rate calculation
                real_interest_rate = nominal_interest_rate - inflation_rate
                st.write("Real Interest Rate:", real_interest_rate)

        elif fin == "Effective Interest Rate":
            st.subheader("Effective Interest Rate Calculator")
            # Input fields for effective interest rate calculation
            nominal_interest_rate = st.number_input("Enter the nominal interest rate (%)")
            compounding_frequency = st.number_input("Enter the compounding frequency (per year)")

            # Calculate button
            if st.button("Calculate"):
                # Perform effective interest rate calculation
                effective_interest_rate = ((1 + (nominal_interest_rate / (100 * compounding_frequency))) ** compounding_frequency - 1) * 100
                st.write("Effective Interest Rate:", effective_interest_rate)
    
        elif fin == "Fixed Interest Rate":
            st.subheader("Fixed Interest Rate")
            # Input fields for fixed interest rate calculation
            initial_interest_rate = st.number_input("Enter the initial interest rate (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform fixed interest rate calculation
                fixed_interest_rate = initial_interest_rate
                st.write("Fixed Interest Rate:", fixed_interest_rate)
        
        elif fin == "Variable Interest Rate":
            st.subheader("Variable Interest Rate")
            # Input fields for variable interest rate calculation
            initial_interest_rate = st.number_input("Enter the initial interest rate (%)")
            variance = st.number_input("Enter the variance (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform variable interest rate calculation
                variable_interest_rate = initial_interest_rate + variance
                st.write("Variable Interest Rate:", variable_interest_rate)
                
        elif fin == "Floating Interest Rate":
            st.subheader("Floating Interest Rate")
            # Input fields for floating interest rate calculation
            benchmark_rate = st.number_input("Enter the benchmark interest rate (%)")
            spread = st.number_input("Enter the spread (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform floating interest rate calculation
                floating_interest_rate = benchmark_rate + spread
                st.write("Floating Interest Rate:", floating_interest_rate)
    
    # ---------------------------------------------------
    
    elif finance_calculation == "Loan Calculator":
        
        st.header("Loan Calculator")
        fin = st.selectbox("Select a loan calculation",("Loan Repayment Schedule", "Loan EMI Calculator", "Loan Affordability Calculator", "Loan Interest Cost Calculator", "Loan Comparison Calculator", "Loan Refinance Calculator"))
                                              
        # don't forget to add simple loan------
        
        if fin == "Loan Repayment Schedule":
            # Input fields for loan repayment schedule calculation
            loan_amount = st.number_input("Enter the loan amount")
            interest_rate = st.number_input("Enter the annual interest rate (%)")
            loan_term = st.number_input("Enter the loan term (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan repayment schedule calculation
                # Add calculation logic here
                pass  # Placeholder

        elif fin == "Loan EMI Calculator":
            # Input fields for loan EMI calculation
            loan_amount = st.number_input("Enter the loan amount")
            interest_rate = st.number_input("Enter the annual interest rate (%)")
            loan_term = st.number_input("Enter the loan term (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan EMI calculation
                # Add calculation logic here
                pass  # Placeholder

        elif fin == "Loan Affordability Calculator":
            # Input fields for loan affordability calculation
            monthly_income = st.number_input("Enter the monthly income")
            monthly_expenses = st.number_input("Enter the monthly expenses")
            other_monthly_obligations = st.number_input("Enter other monthly obligations")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan affordability calculation
                # Add calculation logic here
                pass  # Placeholder
        
        elif fin == "Loan Interest Cost Calculator":
            # Input fields for loan interest cost calculation
            loan_amount = st.number_input("Enter the loan amount")
            interest_rate = st.number_input("Enter the annual interest rate (%)")
            loan_term = st.number_input("Enter the loan term (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan interest cost calculation
                # Add calculation logic here
                pass  # Placeholder
            
        elif fin == "Loan Comparison Calculator":
            # Input fields for loan comparison calculation
            loan1_amount = st.number_input("Enter loan 1 amount")
            loan1_interest_rate = st.number_input("Enter loan 1 annual interest rate (%)")
            loan1_term = st.number_input("Enter loan 1 term (in years)")

            loan2_amount = st.number_input("Enter loan 2 amount")
            loan2_interest_rate = st.number_input("Enter loan 2 annual interest rate (%)")
            loan2_term = st.number_input("Enter loan 2 term (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan comparison calculation
                # Add calculation logic here
                pass  # Placeholder
            
        elif fin == "Loan Refinance Calculator":
            # Input fields for loan refinance calculation
            current_loan_amount = st.number_input("Enter current loan amount")
            current_interest_rate = st.number_input("Enter current annual interest rate (%)")
            current_loan_term = st.number_input("Enter current loan term (in years)")

            new_interest_rate = st.number_input("Enter new annual interest rate for refinance (%)")
            new_loan_term = st.number_input("Enter new loan term for refinance (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan refinance calculation
                # Add calculation logic here
                pass  # Placeholder
            
            
        elif fin == "Loan Prepayment Calculator":
            # Input fields for loan prepayment calculation
            current_loan_amount = st.number_input("Enter current loan amount")
            current_interest_rate = st.number_input("Enter current annual interest rate (%)")
            current_loan_term = st.number_input("Enter current loan term (in years)")
            additional_payment = st.number_input("Enter additional payment amount")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan prepayment calculation
                # Add calculation logic here
                pass  # Placeholder
        
        elif fin == "Loan Amortization Calculator":
            # Input fields for loan amortization calculation
            loan_amount = st.number_input("Enter the loan amount")
            interest_rate = st.number_input("Enter the annual interest rate (%)")
            loan_term = st.number_input("Enter the loan term (in years)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan amortization calculation
                # Add calculation logic here
                pass  # Placeholder
        
        elif fin == "Loan to Value (LTV) Calculator":
            # Input fields for LTV calculation
            property_value = st.number_input("Enter the property value")
            loan_amount = st.number_input("Enter the loan amount")

            # Calculate button
            if st.button("Calculate"):
                # Perform LTV calculation
                ltv_ratio = (loan_amount / property_value) * 100
                st.write("Loan to Value (LTV) Ratio:", ltv_ratio)
    
        elif fin == "Debt Service Coverage Ratio (DSCR) Calculator":
            # Input fields for DSCR calculation
            net_operating_income = st.number_input("Enter net operating income")
            total_debt_service = st.number_input("Enter total debt service")

            # Calculate button
            if st.button("Calculate"):
                # Perform DSCR calculation
                dscr_ratio = net_operating_income / total_debt_service
                st.write("Debt Service Coverage Ratio (DSCR):", dscr_ratio)
    
        elif fin == "Loan Origination Fee Calculator":
            # Input fields for loan origination fee calculation
            loan_amount = st.number_input("Enter the loan amount")
            origination_fee_rate = st.number_input("Enter the origination fee rate (%)")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan origination fee calculation
                origination_fee = (origination_fee_rate / 100) * loan_amount
                st.write("Loan Origination Fee:", origination_fee)
    
        elif fin == "Loan APR Calculator":
            # Input fields for loan APR calculation
            loan_amount = st.number_input("Enter the loan amount")
            annual_interest_rate = st.number_input("Enter the annual interest rate (%)")
            loan_term = st.number_input("Enter the loan term (in years)")
            loan_points = st.number_input("Enter the loan points")
            other_fees = st.number_input("Enter other fees")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan APR calculation
                # Add calculation logic here
                pass  # Placeholder   
    
        elif fin == "Loan Points Calculator":
            # Input fields for loan points calculation
            loan_amount = st.number_input("Enter the loan amount")
            loan_points = st.number_input("Enter the loan points")

            # Calculate button
            if st.button("Calculate"):
                # Perform loan points calculation
                # Add calculation logic here
                pass  # Placeholder
    
    
    
    # ---------------------------------------------------
    
    elif finance_calculation == "Loan Payment Schedule":
        
        def calculate_amortization_schedule(loan_amount, interest_rate, loan_term):
            # Convert annual interest rate to monthly interest rate
            monthly_interest_rate = interest_rate / 12 / 100
            
            # Calculate monthly payment using the formula for loan amortization
            monthly_payment = loan_amount * (monthly_interest_rate / (1 - (1 + monthly_interest_rate) ** (-loan_term * 12)))
            
            # Initialize lists to store schedule data
            periods = []
            interest_payments = []
            principal_payments = []
            total_payments = []
            remaining_balances = []

            # Initialize loan balance
            remaining_balance = loan_amount

            # Loop through each month to calculate schedule
            for month in range(1, loan_term * 12 + 1):
                # Calculate interest payment
                interest_payment = remaining_balance * monthly_interest_rate
                
                # Calculate principal payment
                principal_payment = monthly_payment - interest_payment
                
                total_payment = interest_payment + principal_payment
                
                # Update remaining balance
                remaining_balance -= principal_payment
                
                # Append data to lists
                periods.append(month)
                interest_payments.append(interest_payment)
                principal_payments.append(principal_payment)
                total_payments.append(total_payment)
                remaining_balances.append(remaining_balance)
                
            # Create DataFrame to store schedule
            schedule_df = pd.DataFrame({
                "Payment Number": periods,
                "Interest Payment": interest_payments,
                "Principal Payment": principal_payments,
                "Total Payment": total_payments,
                "Remaining Balance": remaining_balances
            })

            return schedule_df
        
        def calculate_emi_schedule(loan_amount, interest_rate, loan_term):
            # Convert annual interest rate to monthly interest rate
            monthly_interest_rate = interest_rate / 12 / 100
            
            # Calculate monthly payment using the formula for EMI
            monthly_payment = (loan_amount * monthly_interest_rate) / (1 - (1 + monthly_interest_rate) ** (-loan_term * 12))
            
            # Initialize lists to store schedule data
            periods = []
            total_payments = []
            interest_payments = []
            principal_payments = []
            remaining_balances = []

            # Initialize loan balance
            remaining_balance = loan_amount

            # Loop through each month to calculate schedule
            for month in range(1, loan_term * 12 + 1):
                # Calculate interest payment
                interest_payment = remaining_balance * monthly_interest_rate
                
                # Calculate principal payment
                principal_payment = monthly_payment - interest_payment
                
                # Update remaining balance
                remaining_balance -= principal_payment
                
                # Append data to lists
                periods.append(month)
                total_payments.append(monthly_payment)
                interest_payments.append(interest_payment)
                principal_payments.append(principal_payment)
                remaining_balances.append(remaining_balance)
                
            # Create DataFrame to store schedule
            schedule_df = pd.DataFrame({
                "Payment Number": periods,
                "Total Payment": total_payments,
                "Interest Payment": interest_payments,
                "Principal Payment": principal_payments,
                "Remaining Balance": remaining_balances
            })

            return schedule_df
        
        
        def calculate_bullet_payment_schedule(loan_amount, interest_rate, loan_term):
            # Calculate interest accrued over the loan term
            total_interest = loan_amount * interest_rate * loan_term / 100
            
            # Calculate the total payment (principal + interest) at the end of the loan term
            total_payment = loan_amount + total_interest
            
            # Create DataFrame to store schedule
            schedule_df = pd.DataFrame({
                "Payment Number": [1],  # Only one payment at the end
                "Total Payment": [total_payment],
                "Interest Payment": [total_interest],
                "Principal Payment": [loan_amount],
                "Remaining Balance": [0]
            })
            
            return schedule_df
        
        
        def calculate_interest_only_schedule(loan_amount, interest_rate, interest_only_period):
            # Calculate interest payment for the interest-only period
            interest_payment = loan_amount * interest_rate * interest_only_period / 100
            
            # Create DataFrame to store schedule for interest-only period
            schedule_df = pd.DataFrame({
                "Payment Number": range(1, interest_only_period + 1),  # Payment numbers for each month of interest-only period
                "Total Payment": [interest_payment],
                "Interest Payment": [interest_payment],
                "Principal Payment": [0],
                "Remaining Balance": [loan_amount]
            })
            
            return schedule_df
           
           
        def calculate_balloon_payment_schedule(loan_amount, interest_rate, loan_term):
            # Calculate interest accrued over the loan term (may use)
            # total_interest = loan_amount * interest_rate * loan_term / 100
            
            # Calculate the remaining principal balance after the loan term
            remaining_principal = 0  # Since it's a balloon payment, the remaining balance is 0
            
            # Calculate the balloon payment (may use)
            balloon_payment = loan_amount * (1 + (interest_rate / 100 * loan_term))
            
            # Create DataFrame to store schedule
            schedule_df = pd.DataFrame({
                "Payment Number": [1],  # Since there's only one payment for a balloon loan
                "Total Payment": [balloon_payment + interest_rate],
                "Interest Payment": [interest_rate],
                "Principal Payment": [loan_amount],
                "Remaining Balance": [remaining_principal]
            })
            
            return schedule_df
    

        def calculate_fixed_principal_schedule(loan_amount, interest_rate, loan_term):
            # Calculate monthly interest rate
            monthly_interest_rate = interest_rate / 100 / 12
            
            # Calculate total number of payments (months)
            total_payments = loan_term * 12
            
            # Calculate fixed principal payment amount
            fixed_principal_payment = loan_amount / total_payments
            
            # Initialize lists to store schedule data
            payment_no = []
            total_payment = []
            interest_payment = []
            principal_payment = []
            remaining_balance = []
            
            # Initialize remaining loan balance
            remaining_loan_balance = loan_amount
            
            # Generate repayment schedule
            for payment_num in range(1, total_payments + 1):
                # Calculate interest component of payment
                interest_payment.append(remaining_loan_balance * monthly_interest_rate)
                
                # Calculate principal component of payment
                principal_payment.append(fixed_principal_payment)
                
                # Update remaining loan balance
                remaining_loan_balance -= fixed_principal_payment
                
                # Add data to lists
                payment_no.append(payment_num)
                total_payment.append(interest_payment[-1] + principal_payment[-1])
                remaining_balance.append(remaining_loan_balance)
            
            # Create DataFrame for schedule
            schedule_df = pd.DataFrame({
                "Payment Number": payment_no,
                "Total Payment": total_payment,
                "Interest Payment": interest_payment,
                "Principal Payment": principal_payment,
                "Remaining Balance": remaining_balance
            })
            
            return schedule_df
        
        
        def calculate_variable_payment_schedule(loan_amount, interest_rate, loan_term):
            # Calculate monthly interest rate
            monthly_interest_rate = interest_rate / 100 / 12
            
            # Calculate total number of payments (months)
            total_payments = loan_term * 12
            
            # Initialize lists to store schedule data
            payment_no = []
            total_payment = []
            interest_payment = []
            principal_payment = []
            remaining_balance = []
            
            # Initialize remaining loan balance
            remaining_loan_balance = loan_amount
            
            # Function to calculate payment amount for each period
            def calculate_payment(payment_num):
                # Custom logic to determine payment amount (e.g., based on borrower's discretion)
                # This is where you can implement your own logic for variable payments
                return 1000  # Placeholder value
                
            # Generate repayment schedule
            for payment_num in range(1, total_payments + 1):
                # Calculate payment amount for the current period
                payment_amount = calculate_payment(payment_num)
                
                # Calculate interest component of payment
                interest_payment.append(remaining_loan_balance * monthly_interest_rate)
                
                # Calculate principal component of payment (total payment - interest)
                principal_payment.append(payment_amount - interest_payment[-1])
                
                # Update remaining loan balance
                remaining_loan_balance -= principal_payment[-1]
                
                # Add data to lists
                payment_no.append(payment_num)
                total_payment.append(payment_amount)
                remaining_balance.append(remaining_loan_balance)
            
            # Create DataFrame for schedule
            schedule_df = pd.DataFrame({
                "Payment Number": payment_no,
                "Total Payment": total_payment,
                "Interest Payment": interest_payment,
                "Principal Payment": principal_payment,
                "Remaining Balance": remaining_balance
            })
            
            return schedule_df
        
        
        
        # Function to plot loan repayment schedule
        def plot_repayment_schedule(schedule_df):
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Payment Number')
            ax1.set_ylabel('Payment Amount', color=color)
            ax1.plot(schedule_df['Payment Number'], schedule_df['Total Payment'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('Remaining Balance', color=color)  
            ax2.plot(schedule_df['Payment Number'], schedule_df['Remaining Balance'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  
            plt.title('Loan Repayment Schedule')
            st.pyplot(fig)

            # Plot interests paid and unpaid
            plt.figure(figsize=(10, 5))
            plt.bar(schedule_df['Payment Number'], schedule_df['Interest Payment'], color='blue', label='Interest Paid')
            plt.bar(schedule_df['Payment Number'], schedule_df['Total Payment'] - schedule_df['Interest Payment'], 
                    bottom=schedule_df['Interest Payment'], color='orange', label='Principal Paid')
            plt.xlabel('Payment Number')
            plt.ylabel('Amount')
            plt.title('Interests Paid and Unpaid Over Payment Numbers')
            plt.legend()
            st.pyplot(plt)


        
        st.header("Loan Payment Schedule")


        # Selectbox for choosing repayment method
        repayment_methods = ["Amortization", "Equal Installment Method (EMI)", "Bullet Payment",
                            "Interest-Only Payments", "Balloon Payment", "Fixed Principal Payments",
                            "Variable Payments"]
        selected_method = st.selectbox("Select Repayment Method", repayment_methods)
        
        # Input fields for loan parameters
        loan_amount = st.number_input("Enter the loan amount")
        interest_rate = st.number_input("Enter the annual interest rate (%)")
        loan_term = st.number_input("Enter the loan term (in years)")

        loan_term = int(loan_term)

        # Button to calculate loan repayment schedule
        if st.button("Calculate Schedule"):
            if selected_method == "Amortization":
                st.write("""Amortization Schedule:
Repayment involves fixed monthly payments consisting of both principal and interest components.
Each monthly payment contributes towards reducing the outstanding loan balance.
The total payment remains constant, but the portion allocated to principal gradually increases over time.
A typical repayment method for mortgages and installment loans.""")
                
                schedule_df = calculate_amortization_schedule(loan_amount, interest_rate, loan_term)
                
            elif selected_method == "Equal Installment Method (EMI)":
                st.write("""Equal Installment Method (EMI):
Similar to amortization, but the calculation of the monthly payment differs slightly.
The total payment each month remains fixed throughout the loan term.
Includes both principal and interest components, with the interest amount decreasing gradually as the principal balance reduces.
Commonly used for personal loans and auto loans.""")
                
                schedule_df = calculate_emi_schedule(loan_amount, interest_rate, loan_term)
                
            elif selected_method == "Bullet Payment":
                st.write("""Bullet Payment:
Involves a single, large payment (bullet payment) at the end of the loan term.
Typically used in situations where borrowers expect to have sufficient funds to repay the entire loan balance at once.
No periodic payments during the loan term, only one payment at the end.""")
                
                schedule_df = calculate_bullet_payment_schedule(loan_amount, interest_rate, loan_term)
            
            elif selected_method == "Interest-Only Payments":
                st.write("""Interest-Only Payments:
Borrowers make only interest payments during an initial period, usually followed by regular amortizing payments.
The principal balance remains unchanged during the interest-only period.
Commonly used in real estate financing, especially for commercial properties.""")
            
                schedule_df = calculate_interest_only_schedule(loan_amount, interest_rate, loan_term)
            
            elif selected_method == "Balloon Payment":
                with st.expander("Details"):
                    st.write("""Balloon Payment:
        Similar to bullet payment but involves a single large payment (balloon payment) that exceeds the scheduled periodic payments.
        Usually applied to loans with relatively short terms, where borrowers make smaller periodic payments with a large final payment.
        Often used in real estate financing and certain types of business loans.

        The formula `balloon_payment = loan_amount * (1 + (interest_rate / 100 * loan_term))` is used to calculate the balloon payment for a loan. In this formula:

        - `loan_amount` is the initial amount of the loan.
        - `interest_rate` is the annual interest rate for the loan.
        - `loan_term` is the term of the loan in years.

        The formula calculates the balloon payment by adding the accrued interest over the loan term to the original loan amount. The interest is calculated using the formula `(interest_rate / 100 * loan_term)` to determine the total interest accrued over the loan term. This interest amount is added to the original loan amount to find the total balloon payment due at the end of the loan term.

        However, it's important to note that this formula is just an example and may not represent the actual calculation used in all cases. Balloon payments can vary depending on the terms negotiated between the borrower and the lender, so this formula provides a simplified illustration of how a balloon payment might be calculated.

        Balloon Payment: Similar to bullet payment but involves a single large payment (balloon payment) that exceeds the scheduled periodic payments. Usually applied to loans with relatively short terms, where borrowers make smaller periodic payments with a large final payment. Often used in real estate financing and certain types of business loans.""")
                
                schedule_df = calculate_balloon_payment_schedule(loan_amount, interest_rate, loan_term)
            
            elif selected_method == "Fixed Principal Payments":
                st.write("""Fixed Principal Payments:
Involves equal principal payments over the loan term, with varying interest amounts.
Principal payments remain constant, resulting in decreasing total payments over time.
Less common but can be suitable for certain types of loans where borrowers prefer consistent principal reduction.""")

                schedule_df = calculate_fixed_principal_schedule(loan_amount, interest_rate, loan_term)
            
            elif selected_method == "Variable Payments":
                st.write("""Variable Payments:
Payment amounts vary over time based on predefined criteria, such as changes in income or business performance.
Allows flexibility in managing cash flows, with payments adjusted periodically according to specific terms.
Suited for borrowers with fluctuating income or those seeking more flexible repayment options.""")
                                
                schedule_df = calculate_variable_payment_schedule(loan_amount, interest_rate, loan_term)

            # Display loan repayment schedule as table
            st.write(schedule_df)

            # Plot loan repayment schedule
            plot_repayment_schedule(schedule_df)

            
    
    
    # -------------------------------------------------
    # if finance_calculation == "Simple Interest":
    #     st.subheader("Simple Interest")
    #     principal = st.number_input("Principal amount:")
    #     rate = st.number_input("Interest rate (%):") / 100
    #     time = st.number_input("Time period (years):")
    #     interest = principal * rate * time
    #     total_amount = principal + interest
    #     st.write(f"Interest: {interest}")
    #     st.write(f"Total amount: {total_amount}")

    # elif finance_calculation == "Currency Converter":
    #     st.subheader("Currency Converter")
    #     amount = st.number_input("Amount:")
    #     currency_from = st.selectbox("From currency:", ("USD", "EUR", "GBP"))
    #     currency_to = st.selectbox("To currency:", ("USD", "EUR", "GBP"))

    #     # Conversion rates (just placeholders)
    #     conversion_rates = {"USD": {"USD": 1, "EUR": 0.85, "GBP": 0.73},
    #                         "EUR": {"USD": 1.18, "EUR": 1, "GBP": 0.86},
    #                         "GBP": {"USD": 1.38, "EUR": 1.16, "GBP": 1}}

    #     if currency_from != currency_to:
    #         converted_amount = amount * conversion_rates[currency_from][currency_to]
    #         st.write(f"Converted amount: {converted_amount} {currency_to}")
    #     else:
    #         st.write("No conversion needed. Same currency.")
    
    # -------------------------------------------------------------

    elif finance_calculation == "Payback Period":
        # Function to calculate Payback Period
        def calculate_payback_period(initial_investment, cash_flows):
            cumulative_cash_flow = 0
            payback_period = 0
            for i, cash_flow in enumerate(cash_flows):
                cumulative_cash_flow += cash_flow
                if cumulative_cash_flow >= initial_investment:
                    payback_period = i + (initial_investment - (cumulative_cash_flow - cash_flow)) / cash_flow
                    return payback_period
                    
                
            if cumulative_cash_flow < initial_investment:
                st.error("Payback period cannot be calculated as the cash flows have not yet recovered the initial investment.")
                return None
            
            
        def converter(period,payback_period):
            if period =="Yearly":
                years,months,days,hours,minutes=0,0,0,0,0
                # Convert payback period to years, months, days, hours, and minutes
                years = int(payback_period)
                months = int((payback_period - years) * 12)
                days = int(((payback_period - years) * 12 - months) * 30)  # Assuming 30 days per month
                hours = int((((payback_period - years) * 12 - months) * 30 - days) * 24)
                minutes = int(((((payback_period - years) * 12 - months) * 30 - days) * 24 - hours) * 60)

                return years, months, days, hours, minutes
            
            if period =="Quarterly":
                years,Quarters,months,days,hours,minutes=0,0,0,0,0,0
                Quarters = int(payback_period)
                if Quarters >= 4:
                    years = int(Quarters/4)
                Quarters = Quarters % 4
                months = int((payback_period - Quarters) * 3)
                days = int(((payback_period - Quarters) * 3 - months) * 30)  # Assuming 30 days per month
                hours = int((((payback_period - Quarters) * 3 - months) * 30 - days) * 24)
                minutes = int(((((payback_period - Quarters) * 3 - months) * 30 - days) * 24 - hours) * 60)

                return years, Quarters, months, days, hours, minutes
            
            elif period=="Monthly":
                years,months,days,hours,minutes=0,0,0,0,0
                months = int(payback_period)  # 5.65 --> months = 5
                if months >= 12:   # If more than a year, convert to years
                    years = int(months/12)
                months = months % 12
                days = int((payback_period - months) * 30)  # --> 5.65 - 5 =.65 * 30 = 19.5 --> int of this--> 19 remains .5
                hours =  int(((payback_period - months) * 30 - days) * 24)
                minutes = int((((payback_period - months) * 30 - days) * 24 - hours) * 60 )
                
                return years, months, days, hours, minutes
                       
            elif period== "Daily":
                years,months,days,hours,minutes=0,0,0,0,0
                days = int(payback_period)
                if days >= 30:
                    months = int(days/30)
                    if months >= 12:
                        years = int(months/12)
                        months = months%12
                days = days%30
                hours = int((payback_period-days)*24)
                minutes = int(((payback_period-days)*24 - hours ) * 60)
                
                return years,months,days,hours,minutes

            elif period== "Hourly":
                years,months,days,hours,minutes=0,0,0,0,0
                hours = int(payback_period)
                if hours >= 24:
                    days = int(hours/24)
                    if days >= 30:
                        months = int(days/30)
                        days = days%30
                        if months >= 12:
                            years = int(months/12)
                            months = months%12
                hours = hours%24
                minutes = int((payback_period - hours)*60)
                
                return years,months,days,hours,minutes
            
        
        st.subheader("Payback Period")
        initial_investment = st.number_input("Initial Investment:", min_value=0.0, step=1.0)
        time_period = st.selectbox("Time Period of Cash Flows:", ("Yearly", "Monthly", "Quarterly", "Daily", "Hourly"))
        cash_flows_input = st.text_input("Cash Flows (comma-separated):")
        calculate_button = st.button("Calculate Payback Period")
        
        if calculate_button and cash_flows_input:
            cash_flows = [float(x.strip()) for x in cash_flows_input.split(",")]
            payback_period = calculate_payback_period(initial_investment, cash_flows)
            if payback_period is not None:
                if time_period == "Yearly":
                    years, months, days, hours, minutes = converter(time_period, payback_period)
                    st.write(f"Payback Period: {years} years {months} months {days} days {hours} hours {minutes} minutes")
                
                elif time_period == "Quarterly":
                    years, Quarters, months, days, hours, minutes = converter(time_period, payback_period)
                    st.write(f"Payback Period: {years} years {Quarters} quarters {months} months {days} days {hours} hours {minutes} minutes")
                   
                elif time_period == "Monthly":
                    years, months, days, hours, minutes = converter(time_period, payback_period)
                    st.write(f"Payback Period: {years} years {months} months {days} days {hours} hours {minutes} minutes")
                   
                elif time_period == "Daily":
                    years, months, days, hours, minutes = converter(time_period, payback_period)
                    st.write(f"Payback Period: {years} years {months} months {days} days {hours} hours {minutes} minutes")
                   
                elif time_period == "Hourly":
                    years, months, days, hours, minutes = converter(time_period, payback_period)
                    st.write(f"Payback Period: {years} years {months} months {days} days {hours} hours {minutes} minutes")
                    
# ------------------------------------------------------------------------------------------------------------
                   

    elif finance_calculation == "Net Present Value (NPV)":
        # Function to calculate Net Present Value (NPV)
        def calculate_npv(initial_investment, cash_flows, discount_rate):
            npv = -initial_investment
            for i, cash_flow in enumerate(cash_flows):
                npv += cash_flow / (1 + discount_rate) ** (i+1)
            return npv
        
        st.subheader("Net Present Value (NPV)")
        initial_investment = st.number_input("Initial Investment:", min_value=0.0, step=1.0)
        discount_rate = st.number_input("Discount Rate (%):") / 100
        time_period = st.selectbox("Time Period of Cash Flows:", ("Yearly", "Monthly", "Quarterly", "Daily"))
        cash_flows_input = st.text_input("Cash Flows (comma-separated):")
        calculate_button = st.button("Calculate NPV")
        if calculate_button and cash_flows_input:
            cash_flows = [float(x.strip()) for x in cash_flows_input.split(",")]
            npv = calculate_npv(initial_investment, cash_flows, discount_rate)
            st.write(f"Net Present Value (NPV): {npv:.2f}")

    
    
    elif finance_calculation == "Internal Rate of Return (IRR)":
        # Function to calculate Internal Rate of Return (IRR)
        def calculate_irr(cash_flows):
            return npf.irr(cash_flows)
    
        st.subheader("Internal Rate of Return (IRR)")
        
        st.write('''Returning nan (not a number) for the provided cash flows?\n
        This could happen if the cash flows do not follow the expected pattern for calculating the IRR. Here are a few common reasons why you might encounter this issue:\n
        1.No change in sign of cash flows: The IRR function requires at least one positive and one negative cash flow for calculation. Ensure that your cash flows alternate between positive and negative values.\n
        2.Too few or too many cash flows: The IRR function may not work well if there are too few cash flows or if there are too many cash flows. Try to provide a reasonable number of cash flows for calculation.\n
        3.Cash flows are too small or too large: Extremely small or large cash flows may lead to numerical instability and cause the IRR function to return nan. Try to keep the magnitude of cash flows within a reasonable range.\n
        4.Cash flows are all zeros: If all cash flows are zeros, the IRR cannot be calculated. Ensure that there is at least one non-zero cash flow for calculation.''')

        
        time_period = st.selectbox("Time Period of Cash Flows:", ("Yearly", "Monthly", "Quarterly", "Daily"))
        cash_flows_input = st.text_input("Cash Flows (comma-separated):")
        calculate_button = st.button("Calculate IRR")
        
        if calculate_button and cash_flows_input:
            cash_flows = [float(x.strip()) for x in cash_flows_input.split(",")]
            irr = calculate_irr(cash_flows)
            st.write(f"Internal Rate of Return (IRR): {irr:.2%}")     
            
            
            
            
            
            
            
           
            
# ----------------------------------------------------------FRIENDS / HELPING SETS----------------------------------------------------------------------------------------
    
    

# ------------------------------------------------------------------------------------------------------------------



# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with :heart: by SAMBIT KUMAR NAYAK")
st.sidebar.markdown("contact: sambitnayak.tuserate@gmail.com")
