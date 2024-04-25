import express from 'express';
import { createServer } from 'vite';
import { spawnSync } from 'child_process';
import path from 'path';
import morgan from 'morgan';
import { PineconeClient } from "@pinecone-database/pinecone";


const app = express();
const port = 3000;
// const port = 3001;
const PYTHON = process.env.PYTHON_PATH;

// const py_script_path = 'pycodes/script1.py';

// Use morgan middleware with 'dev' format
app.use(morgan('dev'));

app.use(express.json());

// Define an Express route that interacts with the Pinecone API
app.get('/api/getNamespaces', async (req, res) => {
  const pinecone = new PineconeClient();
  await pinecone.init({
    environment: "us-east-1-aws",
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pinecone.Index("pinecone-test");
  const indexStats = await index.describeIndexStats({
    describeIndexStatsRequest: {
      filter: {},
    },
  });

  let jsonObj = indexStats.namespaces;  // Get the namespaces from Pinecone
  let options = Object.keys(jsonObj);
  res.json(options);  // Send the namespaces to the client
});


/* Use the snippet below to use fixed python file path */
// app.post('/api/runPythonScript', (req, res) => {
//   const result = callSync(PYTHON, path.join(process.cwd(), py_script_path), req.body);
//   res.json({ result: result });
// });

/* Use the snippet below to use custom python file path */
app.post('/api/runPythonScript', (req, res) => {
  const { py_script_path, ...args } = req.body;

  const result = callSync(
    PYTHON,
    path.join(process.cwd(), py_script_path),
    args
  );

  res.json({ result: result });
});



function callSync(PYTHON, script, args, options) {
  const result = spawnSync(PYTHON, [script, JSON.stringify(args)]);
  if (result.status === 0) {
    const ret = result.stdout.toString();
    if (!(ret instanceof Error)) {
      return ret;
    } else {
      return {
        error: ret.message
      };
    }
  } else if (result.status === 1) {
    return {
      error: result.stderr.toString()
    };
  } else {
    return {
      error: result.stdout.toString()
    };
  }
}

async function createExpressServer() {
  const vite = await createServer({
    server: { middlewareMode: true },
  });

  app.use(vite.middlewares);

  app.listen(port, () => {
    console.log(`Express server started on http://localhost:${port}!`);
  });
}

createExpressServer();
