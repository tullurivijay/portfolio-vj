export interface Architecture {
    id: number;
    title: string;
    projectName: string;
    description: string;
    techStack: string[];
    imagePath: string;
    category: 'Data Engineering' | 'Machine Learning' | 'MLOps' | 'Infrastructure';
    highlights: string[];
}

export const architectures: Architecture[] = [
    {
        id: 1,
        title: "Agentic AI Sustainability Assistant Architecture",
        projectName: "Nike Sustainability Foundation",
        description: "LangChain-powered RAG system retrieving carbon emissions metrics from Databricks vector store for regulatory-compliant sustainability reporting. Features multi-agent orchestration and real-time data retrieval.",
        techStack: ["LangChain", "RAG", "Databricks", "Vector Store", "FastAPI", "Python"],
        imagePath: "/architecture/agentic-ai-architecture.png",
        category: "Machine Learning",
        highlights: [
            "Multi-agent LangChain orchestration",
            "Databricks vector store integration",
            "Real-time carbon metrics retrieval",
            "Regulatory compliance automation"
        ]
    },
    {
        id: 2,
        title: "Enterprise FAE Metrics Platform Architecture",
        projectName: "Fidelity Investments",
        description: "Multi-source data ingestion platform aggregating metrics from SonarQube, GitHub, Jenkins, and Jira into Delta Lake. Implements medallion architecture with automated KPI computation and leadership dashboards.",
        techStack: ["Delta Lake", "PySpark", "Apache Airflow", "SonarQube", "GitHub API", "Jenkins", "Jira API"],
        imagePath: "/architecture/fae-metrics-architecture.png",
        category: "Data Engineering",
        highlights: [
            "Multi-source data aggregation (4 platforms)",
            "Medallion architecture (Bronze/Silver/Gold)",
            "Automated KPI dashboard delivery",
            "200+ repository monitoring"
        ]
    },
    {
        id: 3,
        title: "Hybrid Cloud Airflow Infrastructure",
        projectName: "Apple Ad-Platforms",
        description: "Enterprise-scale Apache Airflow deployment across 17 environments (on-prem and AWS). Terraform-based IaC enabling scalable DAG executions with Kubernetes orchestration.",
        techStack: ["Apache Airflow", "Terraform", "AWS", "Kubernetes", "Docker", "GitOps"],
        imagePath: "/architecture/airflow-infrastructure-architecture.png",
        category: "Infrastructure",
        highlights: [
            "17 environment deployment",
            "Hybrid cloud architecture (on-prem + AWS)",
            "Terraform IaC automation",
            "Kubernetes-based scaling"
        ]
    },
    {
        id: 4,
        title: "Real-Time Fraud Detection Engine",
        projectName: "Capital One",
        description: "Streaming data pipeline orchestrating data from Snowflake, Data Lake, and Elasticsearch into OpenML-based fraud risk engine. Enables real-time fraud score computation with sub-second latency.",
        techStack: ["OpenML", "Snowflake", "Elasticsearch", "Apache Airflow", "Flask", "AWS"],
        imagePath: "/architecture/fraud-detection-architecture.png",
        category: "Machine Learning",
        highlights: [
            "Real-time fraud scoring",
            "Multi-source data orchestration",
            "Sub-second latency",
            "OpenML model serving"
        ]
    },
    {
        id: 5,
        title: "Medallion Architecture Migration",
        projectName: "Nike Sustainability Foundation",
        description: "Snowflake to Databricks migration implementing medallion architecture with Z-ordering and liquid clustering. Features raw, bronze, silver, and gold layers with automated data quality checks.",
        techStack: ["Databricks", "Delta Lake", "Snowflake", "PySpark", "Unity Catalog"],
        imagePath: "/architecture/medallion-architecture.png",
        category: "Data Engineering",
        highlights: [
            "Snowflake to Databricks migration",
            "4-layer medallion architecture",
            "Z-ordering & liquid clustering",
            "Automated data quality checks"
        ]
    },
];

export const architectureCategories = [
    "All",
    "Data Engineering",
    "Machine Learning",
    "MLOps",
    "Infrastructure"
] as const;
