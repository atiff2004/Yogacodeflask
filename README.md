# Yogacodeflask
Step 1: Extract the Flask Backend Code
->Download and extract the zip file containing the Flask backend code.
->Navigate to the extracted folder using your command prompt.
->cd path\to\extracted\flask-folder / (or open extract folder in cmd)


Step 2: Set Up a Virtual Environment
Step 3: Install Dependencies (requirements.txt)
 ->pip install -r requirements.txt
Step 4: Deploy the Flask App
Step 5: Verify the API Endpoint
 ->Ensure that the Flask API endpoint (e.g., https://your-app-name.server.com/video_feed) 
   is up and running.

 -----------------Setting Up the React Native Project-------------------------
Step 1: Clone the React Native Repository
 ->git clone https://github.com/UmerAnsari222/video_stream_client_demo.git
     cd video_stream_client_demo
Step 2: Install React Native Dependencies
 ->yarn install
Step 3: Update the API URL
In the React Native project, update the API URL to point to the deployed Flask backend:
 ->const API_URL = 'https://your-app-name.server.com/video_feed';
Step 4: Run the React Native App
 ->Ensure your development environment is set up (Android Studio)
->yarn android
