import streamlit as st
import pandas as pd
import nlpcloud

# Streamlit page config
st.set_page_config(page_title="Review Sumamrizer")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ("Home", "Dashboard"))

# Streamlit app content based on page selection
if page == "Home":
    # Main content of first.py
    st.title("Product Reviews Search App")

    # Loading product data
    product_data = pd.read_csv('Cleaned_Reviews.csv')
    product_data['Brand'] = product_data['Brand'].str.lower()
    product_data['Category'] = product_data['Category'].str.lower()

    # Initialize NLPCloud Clients (API key should ideally be in secrets.toml)
    summarization_client = nlpcloud.Client("finetuned-llama-3-70b", "1deece16716f75baa63f911d0e73ededa321d429", gpu=True)
    sentiment_client = nlpcloud.Client("distilbert-base-uncased-finetuned-sst-2-english", "1deece16716f75baa63f911d0e73ededa321d429", gpu=False)

    # Search functionality
    search_term = st.text_input("Search for a Brand or Category", "").lower()

    if search_term:
        filtered_data = product_data[(product_data['Brand'].str.contains(search_term)) | 
                                     (product_data['Category'].str.contains(search_term))]

        if not filtered_data.empty:
            st.write(f"Showing results for: {search_term}")
            filtered_display = filtered_data[['Brand', 'Category', 'Reviews', 'Price']]
            st.write(filtered_display)
            

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

                
                if reviews_text:
                    summary = summarization_client.summarization(reviews_text, size="small")['summary_text']
                    st.write(f"Summary of Reviews for {selected_product}:")
                    st.write(summary)

                    sentiment_response = sentiment_client.sentiment(summary)['scored_labels']
                    sentiment_analysis = [(item['label'], round(item['score'], 4)) for item in sentiment_response]
                    st.write(f"Sentiment Analysis for {selected_product}:")
                    for sentiment in sentiment_analysis:
                        st.write(f"{sentiment[0]}: {sentiment[1]}")
                else:
                    st.write(f"No reviews found for the selected product: {selected_product}")
        else:
            st.write("No results found for your search.")

elif page == "Dashboard":
    # Now load the content from dashboard.py (You can import functions from dashboard.py if necessary)
    st.title("Dashboard Page")
    st.write("This is the Dashboard content.")
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
    
        # Add the actual code or import functions from dashboard.py if needed.
        # For example:
        # from dashboard import display_dashboard
        # display_dashboard()
