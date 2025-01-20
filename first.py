import streamlit as st
import pandas as pd
import nlpcloud

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)


st.sidebar.success("Select a page above.")




product_data = pd.read_csv(r'C:\Users\shiva\OneDrive\Desktop\kpmg\Cleaned_Reviews.csv')

product_data['Brand'] = product_data['Brand'].str.lower()
product_data['Category'] = product_data['Category'].str.lower()


# Initialize NLPCloud Clients
summarization_client = nlpcloud.Client("finetuned-llama-3-70b", "3b4fe4d06f3b1ef2754d8b8afdc51661e3decdf5", gpu=True)
sentiment_client = nlpcloud.Client("distilbert-base-uncased-emotion", "3b4fe4d06f3b1ef2754d8b8afdc51661e3decdf5", gpu=False)


# Streamlit app code
st.title("Product Reviews Search App")

# Creating a search bar
search_term = st.text_input("Search for a Brand or Category", "")

# Lowercase the search term for case-insensitive matching
search_term = search_term.lower()

# When user types in search term, filter the dataset
if search_term:
    # Filter by brand or category
    filtered_data = product_data[(product_data['Brand'].str.contains(search_term)) | 
                                 (product_data['Category'].str.contains(search_term))]

    if not filtered_data.empty:
        # Display filtered results, limited to 10 entries
        st.write(f"Showing results for: {search_term}")
        filtered_display = filtered_data[['Brand', 'Category', 'Reviews', 'Price']]
        st.write(filtered_display)

        # Choose a specific product to analyze from the filtered data
        selected_product = st.selectbox("Select a product to analyze", filtered_display['Brand'].unique())

        if selected_product:
            # Get the reviews for the selected product
            product_reviews = filtered_data[filtered_data['Brand'] == selected_product]['Reviews'].dropna().tolist()
            reviews_text = " ".join(product_reviews)

            if reviews_text:
                # Summarize the reviews
                summary = summarization_client.summarization(reviews_text, size="small")['summary_text']
                st.write(f"Summary of Reviews for {selected_product}:")
                st.write(summary)

                # Sentiment analysis
                sentiment_response = sentiment_client.sentiment(summary)['scored_labels']
                sentiment_analysis = [(item['label'], round(item['score'], 4)) for item in sentiment_response]

                # Display sentiment analysis results
                st.write(f"Sentiment Analysis for {selected_product}:")
                for sentiment in sentiment_analysis:
                    st.write(f"{sentiment[0]}: {sentiment[1]}")
            else:
                st.write(f"No reviews found for the selected product: {selected_product}")
    else:
        st.write("No results found for your search.")
else:
    st.write("Please enter a search term to find relevant products.")

