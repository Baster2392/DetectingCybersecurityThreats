# Cyber Threat Detection Using Deep Learning

## Project Overview

As cyber threats continue to grow in both sophistication and frequency, organizations worldwide face an urgent need to adopt advanced security measures. Traditional threat detection methods often fall short in adapting to new and evolving threats. This project aims to leverage deep learning models to detect cyber threats more effectively.

The goal is to design and implement a deep learning model that can identify cyber threats using simulated real-world log data from the BETH dataset. By successfully developing this model, you will contribute to enhancing cybersecurity measures, helping organizations safeguard their sensitive information and ensure operational continuity.

## Data

The BETH dataset provides the following columns:

| Column            | Description                                                                  |
|-------------------|------------------------------------------------------------------------------|
| `processId`       | Unique identifier for the process that generated the event - `int64`         |
| `threadId`        | ID for the thread spawning the log - `int64`                                 |
| `parentProcessId` | Label for the process spawning this log - `int64`                            |
| `userId`          | ID of the user spawning the log                                              |
| `mountNamespace`  | Mounting restrictions the process log works within - `int64`                 |
| `argsNum`         | Number of arguments passed to the event - `int64`                            |
| `returnValue`     | Value returned from the event log (usually 0) - `int64`                      |
| `sus_label`       | Binary label indicating a suspicious event (1 - suspicious, 0 - benign) - `int64` |