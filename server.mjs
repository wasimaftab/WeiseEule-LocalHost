import express from 'express';
import { createServer } from 'vite';
// import { spawnSync } from 'child_process';
// import path from 'path';
import morgan from 'morgan';
// OLD import { PineconeClient } from "@pinecone-database/pinecone"; 
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
const PORT = process.env.PORT || 3000; // Allow port to be set by environment for flexibility

// Use morgan middleware with 'dev' format
app.use(morgan('dev'));

// Parse JSON bodies (as sent by API clients)
app.use(express.json());

// // Define an Express route that interacts with the Pinecone API
/* NEW: Compatible for pinecone v2.2.1 */
app.get('/api/getNamespaces', async (req, res) => {
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
  });

  const index = pinecone.Index(process.env.PINECONE_INDEX);
  const indexStats = await index.describeIndexStats({
    describeIndexStatsRequest: {
      filter: {},
    },
  });

  let jsonObj = indexStats.namespaces;  // Get the namespaces from Pinecone
  let options = Object.keys(jsonObj);
  res.json(options);  // Send the namespaces to the client
});


// Forward API requests to the FastAPI backend
app.post('/api/:endpoint', async (req, res) => {
  const endpoint = req.params.endpoint;
  try {
    // Dynamically build the URL to the FastAPI service
    const response = await fetch(`http://localhost:8000/${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    // Forward the status code from the FastAPI response
    res.status(response.status);

    // Parse and forward the JSON response body
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error forwarding request:', error);
    res.status(500).json({ error: "Error forwarding request to FastAPI" });
  }
});

// Optional: A simple health check endpoint
app.get('/api/health', async (req, res) => {
  res.json({ status: 'OK' });
});

// Function to start the Express server
async function startServer() {
  // In development, integrate Vite's middleware for hot module replacement (HMR)
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'custom',
    });
    app.use(vite.middlewares);
  }
  
  // Start the server
  app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

startServer();
