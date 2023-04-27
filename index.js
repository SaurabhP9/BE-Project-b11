const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const app = express();
const { spawn } = require("child_process");

// Set up middleware to parse JSON request body
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Define endpoint to check the video is okay or not
app.post("/check-id", (req, res) => {
  const id = req.body.id || null;
  console.log("Received ID: " + id);
  if (id) {
    var dataToSend;
    const python = spawn("python", ["./Test.py", id]);

    // collect data from script
    python.stdout.on("data", function (data) {
      dataToSend = data.toString();
    });
    console.log(dataToSend);
    python.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });

    // in close event we are sure that stream from child process is closed
    python.on("exit", (code) => {
      // Send response to the client with the result of the check
      res.status(200).send(dataToSend);
      console.log(`child process exited with code ${code} ` + dataToSend);
    });
  } else {
    res.status(400).send("Error: URL is null or empty.");
  }
});

// Start the server on port 3000
app.listen(3000, () => {
  console.log("Server started on port 3000");
});
