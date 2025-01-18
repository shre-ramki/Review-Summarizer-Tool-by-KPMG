import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
product_data = pd.read_csv(r'Cleaned_Reviews.csv')

# Add title to the dashboard
st.title("Product Reviews Dashboard")

# Show some basic statistics
st.subheader("Basic Statistics")
st.write("Number of products:", len(product_data))
st.write("Number of unique brands:", product_data['Brand'].nunique())
st.write("Number of unique categories:", product_data['Category'].nunique())

# Create a bar chart of the number of reviews by brand


# Create a price distribution plot
st.subheader("Price Distribution")
fig, ax = plt.subplots()
product_data['Price'].hist(bins=20, ax=ax)
ax.set_title("Price Distribution of Products")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)


# Create a rating distribution plot
st.subheader("Rating Distribution")
fig, ax = plt.subplots()
sns.histplot(product_data['Rating'], bins=5, kde=True, ax=ax)
ax.set_title("Rating Distribution")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Create a price vs. rating scatter plot
st.subheader("Price vs. Rating")
fig, ax = plt.subplots()
sns.scatterplot(x='Price', y='Rating', data=product_data, hue='Category', ax=ax)
ax.set_title("Price vs. Rating")
ax.set_xlabel("Price")
ax.set_ylabel("Rating")
st.pyplot(fig)
