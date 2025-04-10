import streamlit as st
import pandas as pd
import nlpcloud

# Streamlit page config
st.set_page_config(page_title="Review Summarizer")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ("Home", "Dashboard"))

# Streamlit app content based on page selection
if page == "Home":
    # Main content of first.py
    st.title("Product Review Summarizer Tool")

    # Loading product data
    product_data = pd.read_csv('Cleaned_Reviews.csv')
    product_data['Brand'] = product_data['Brand'].str.lower()
    product_data['Category'] = product_data['Category'].str.lower()

    # Initialize NLPCloud Clients (API key should ideally be in secrets.toml)
    summarization_client = nlpcloud.Client("finetuned-llama-3-70b", "cf38e55eddad01cbb5cea441c7c49c041ae7f65d", gpu=True)
    sentiment_client = nlpcloud.Client("distilbert-base-uncased-finetuned-sst-2-english", "cf38e55eddad01cbb5cea441c7c49c041ae7f65d", gpu=False)

    # Category selection dropdown
    category_options = product_data['Category'].unique()
    selected_category = st.selectbox("Select a Category", ["All"] + list(category_options))

    if selected_category != "All":
        filtered_data = product_data[product_data['Category'] == selected_category]
    else:
        filtered_data = product_data

    # Removed the search term input
    # filtered_data = filtered_data[(filtered_data['Brand'].str.contains(search_term)) |
    #                               (filtered_data['Category'].str.contains(search_term))]

    if not filtered_data.empty:
        # st.write(f"Showing results in category: {selected_category}")
        filtered_display = filtered_data[['Brand', 'Category', 'Reviews', 'Price']]
        # st.write(filtered_display)

        selected_product = st.selectbox("Select a product to analyze", filtered_display['Brand'].unique())
        if selected_product:
            product_reviews = filtered_data[filtered_data['Brand'] == selected_product]['Reviews'].dropna().tolist()
            reviews_text = " ".join(product_reviews)
            filtered_select_product = filtered_data[filtered_data['Brand'] == selected_product]

            if not filtered_select_product.empty:
                overall_rating = filtered_select_product.iloc[0]['Rating']
                st.write(f"Overall rating for the product {selected_product} is: {overall_rating}")
            else:
                st.write(f"No product found for the selected brand: {selected_product}")

            # Try to get the summary and sentiment analysis
            if reviews_text:
                try:
                    summary = summarization_client.summarization(reviews_text, size="small")['summary_text']
                    st.write(f"Summary of Reviews for {selected_product}:")
                    st.write(summary)

                    sentiment_response = sentiment_client.sentiment(summary)['scored_labels']
                    sentiment_analysis = [(item['label'], round(item['score']*100, 4)) for item in sentiment_response]
                    st.write(f"Sentiment Analysis for {selected_product}:")
                    for sentiment in sentiment_analysis:
                        st.write(f"{sentiment[0]}: {sentiment[1]}%")
                except Exception:
                    # If there is an error, print a custom message
                    st.error("Error occurred. Please try again later.")
            else:
                st.write(f"No reviews found for the selected product: {selected_product}")
    else:
        st.write("No results found for your selection.")
elif page == "Dashboard":
    # Now load the content from dashboard.py (You can import functions from dashboard.py if necessary)
    st.title("Dashboard Page")
    st.write("This is the Dashboard content.")
    
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
    
    # Create a price vs. rating scatter 
