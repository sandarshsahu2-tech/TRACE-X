\# TRACE-X



\## AI-Powered Financial Crime Investigation Platform



TRACE-X is a student-built financial crime investigation prototype designed to turn large volumes of transaction activity into focused, explainable investigation leads.



The core workflow is:



\*\*DETECT → CONNECT → EXPLAIN → INVESTIGATE\*\*



TRACE-X combines:



\- Machine-learning risk detection

\- Historical transaction intelligence

\- Network intelligence

\- Investigation prioritisation

\- Evidence-grounded AI investigation



> TRACE-X decides. AI explains.



The AI layer does not independently determine whether a transaction is criminal. It works from evidence supplied by TRACE-X and turns that evidence into an investigator-readable briefing.



\---



\## Current Prototype



\### TRACE-X V1



\- 38 ordered model features

\- XGBoost-based risk engine

\- 800 boosting rounds

\- Decision threshold: `0.76`

\- Risk output: `risk score + FLAG/NORMAL`



\### Dataset



The project was developed using a large synthetic AML transaction dataset containing:



\*\*6,924,049 transactions\*\*



The raw dataset is intentionally not included in this repository because of its size.



\### Main Interfaces



\- Command Center

\- Investigation Queue

\- Network Intelligence

\- Rule Engine

\- AI Investigation

\- Transaction analysis workflow



\---



\## Architecture



```text

Transaction Data

&#x20;      ↓

Data Processing

&#x20;      ↓

Historical / Behavioural Features

&#x20;      ↓

38-Feature Model Contract

&#x20;      ↓

TRACE-X V1

&#x20;      ↓

Risk Score + Decision

&#x20;      ↓

Historical + Network Evidence

&#x20;      ↓

Investigation Interface

&#x20;      ↓

AI Investigation Explanation

