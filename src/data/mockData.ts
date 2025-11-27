import { Github, Linkedin, Mail, Code, Database, Globe, Server, Terminal, Cloud, GitBranch, Workflow } from 'lucide-react';

export const personalInfo = {
    name: "Vijay Tulluri",
    title: "Senior Software Engineer | Technical Leader - Data & AI",
    description: "IT Professional with 9+ years of experience in Data Engineering, Software Development, and Machine Learning. Expert in building scalable data pipelines, ML systems, and Agentic AI solutions across sustainability, finance, retail, and ad-tech domains.",
    email: "Tulluri.Vijay@gmail.com",
    linkedin: "https://www.linkedin.com/in/vijaytulluri/",
    github: "https://github.com/tullurivijay",
    resume: "#",
};


export const skills = [
    { name: "Python & PySpark", icon: Code, level: "Expert" },
    { name: "Databricks & Delta Lake", icon: Database, level: "Expert" },
    { name: "AWS & Cloud", icon: Cloud, level: "Expert" },
    { name: "Apache Airflow", icon: Workflow, level: "Expert" },
    { name: "LLMs & Transformers", icon: Terminal, level: "Expert" },
    { name: "RAG & LangChain", icon: Code, level: "Expert" },
    { name: "Snowflake", icon: Database, level: "Advanced" },
    { name: "BERT & NLP", icon: Terminal, level: "Advanced" },
    { name: "Machine Learning & MLOps", icon: Terminal, level: "Advanced" },
    { name: "Weights & Biases", icon: GitBranch, level: "Advanced" },
    { name: "Docker & Kubernetes", icon: Server, level: "Advanced" },
    { name: "Terraform & IaC", icon: GitBranch, level: "Advanced" },
];


export const experience = [
    {
        id: 1,
        role: "Sr. Software Engineer – Data Engineering & ML",
        company: "Fidelity Investments, FAE Metrics",
        period: "Aug 2025 – Current",
        description: "Built enterprise-scale data pipelines aggregating SonarQube, GitHub, Jenkins, and Jira metrics into Delta Lake. Leading model-driven MCP framework standardizing code quality across 200+ repos.",
    },
    {
        id: 2,
        role: "Sr. Software Engineer – Data Engineering & ML",
        company: "Nike, Sustainability Foundation Analytics",
        period: "Aug 2023 – Jul 2025",
        description: "Developed Agentic AI-powered sustainability assistant using LangChain with RAG. Built scalable ingestion pipelines from 40+ sources. Led Snowflake to Databricks migration using medallion architecture.",
    },
    {
        id: 3,
        role: "Software Engineer III – Data Engineering & ML",
        company: "Apple, Ad-Platforms",
        period: "Apr 2022 – June 2023",
        description: "Architected and deployed Apache Airflow across 17 environments for Apple's Ad Platforms. Built ETL test automation frameworks reducing manual testing by 60%.",
    },
    {
        id: 4,
        role: "Senior Software Engineer & ML",
        company: "Vanguard, Advice Engagement Labs",
        period: "Dec 2021 – Apr 2022",
        description: "Engineered highly scalable ETL/ELT pipelines with Airflow and Snowflake. Integrated ML recommendation outputs with Microsoft Dynamics 365 using RESTful APIs.",
    },
    {
        id: 5,
        role: "Senior Data Engineer",
        company: "Capital One, Fraud & Risk Mitigation",
        period: "Sep 2019 – Dec 2021",
        description: "Orchestrated data pipelines into OpenML-based fraud risk engine. Integrated DVC with S3 for dataset versioning. Built complex DAGs in Apache Airflow.",
    },
    {
        id: 6,
        role: "Data & AI Research",
        company: "University of North Texas",
        period: "Aug 2017 – May 2019",
        description: "Developed distributed big data pipeline for predicting energy consumption using Apache Spark, XGBoost, and LSTM. Achieved 15% improvement over baseline models.",
    },
    {
        id: 7,
        role: "Software Engineer – Data & AI",
        company: "Invesco, Budget & Forecasting",
        period: "Oct 2015 – July 2017",
        description: "Reduced asset maintenance costs by 30% with real-time anomaly detection using CART and ARIMA. Built streaming pipelines with Kafka and InfluxDB.",
    },
];

export const projects = [
    {
        id: 1,
        title: "Agentic AI Sustainability Assistant",
        description: "Built AI-powered assistant using LangChain with RAG, retrieving carbon emissions metrics from Databricks vector store for regulatory-compliant sustainability reporting.",
        tags: ["LangChain", "RAG", "Databricks", "FastAPI"],
        link: "#",
        github: "#",
    },
    {
        id: 2,
        title: "Enterprise FAE Metrics Platform",
        description: "Aggregated SonarQube, GitHub, Jenkins metrics into Delta Lake to compute engineering KPIs, enabling automated delivery of team health dashboards to leadership.",
        tags: ["Delta Lake", "PySpark", "Airflow"],
        link: "#",
        github: "#",
    },
    {
        id: 3,
        title: "Hybrid Cloud Airflow Infrastructure",
        description: "Architected Apache Airflow across 17 environments (on-prem and AWS) with Terraform IaC, supporting scalable DAG executions for Apple's Ad Platforms.",
        tags: ["Airflow", "Terraform", "AWS", "Kubernetes"],
        link: "#",
        github: "#",
    },
    {
        id: 4,
        title: "Real-Time Fraud Detection Engine",
        description: "Orchestrated pipelines from Snowflake, Data Lake, and Elasticsearch into OpenML-based fraud risk engine, enabling real-time fraud score computation.",
        tags: ["OpenML", "Snowflake", "Flask", "AWS"],
        link: "#",
        github: "#",
    },
    {
        id: 5,
        title: "Smart Campus Energy Forecasting",
        description: "Developed ML pipeline using XGBoost, LSTM, and Prophet for time-series forecasting of campus energy consumption, achieving 15% improvement over baseline.",
        tags: ["Python", "XGBoost", "LSTM", "Spark"],
        link: "#",
        github: "#",
    },
    {
        id: 6,
        title: "Medallion Architecture Migration",
        description: "Led Snowflake to Databricks migration implementing raw, bronze, silver, gold layers with Z-ordering and liquid clustering for optimized query performance.",
        tags: ["Databricks", "Delta Lake", "Snowflake"],
        link: "#",
        github: "#",
    },
];
