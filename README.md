<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README: Setting Up Your Environment with Pipenv</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        
        h3 {
            color: #2980b9;
            margin-top: 25px;
        }
        
        .prerequisite {
            background-color: #f8f9fa;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .step {
            background-color: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .command-group {
            margin-bottom: 20px;
        }
        
        .note {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 15px 0;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>README: Setting Up Your Environment with Pipenv</h1>
    
    <div class="prerequisite">
        <h2>Prerequisite: Install Pipenv</h2>
        <p>Follow the official Pipenv installation guide to set up Pipenv on your system: 
        <a href="https://pipenv.pypa.io/en/latest/installation/" target="_blank">Install Pipenv Documentation</a></p>
    </div>
    
    <div class="step">
        <h2>Steps to Set Up the Environment</h2>
        
        <h3>Install Required Packages</h3>
        <p>Run the following commands in your terminal (assuming Pipenv is already installed):</p>
        
        <div class="command-group">
            <h4>Core ML and AI Dependencies:</h4>
            <pre><code>pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf</code></pre>
        </div>
        
        <div class="command-group">
            <h4>Hugging Face Integration:</h4>
            <pre><code>pipenv install huggingface_hub</code></pre>
        </div>
        
        <div class="command-group">
            <h4>Web Interface Framework:</h4>
            <pre><code>pipenv install streamlit</code></pre>
        </div>
        
        <div class="note">
            <strong>Note:</strong> These commands will create a virtual environment and install all necessary dependencies for your medical chatbot project.
        </div>
        
        <h3>Activate the Environment</h3>
        <p>After installation, activate your Pipenv environment:</p>
        <pre><code>pipenv shell</code></pre>
        
        <h3>Run Your Application</h3>
        <p>Once the environment is activated, you can run your Streamlit application:</p>
        <pre><code>streamlit run app.py</code></pre>
    </div>
</body>
</html>
