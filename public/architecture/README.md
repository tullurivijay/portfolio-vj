# Architecture Diagrams - Instructions

## How to Add Your Eraser.io Diagrams

Your System Design Architecture section is ready! Currently using AI-generated placeholder diagrams. Follow these steps to use your custom Eraser.io diagrams:

### Step 1: Export from Eraser.io

1. Open your Eraser.io workspace: https://app.eraser.io/workspace/8uWXYQQoI3We6eIeOwTF
2. For each diagram you want to include:
   - Click **Export** or **Share**
   - Select **PNG** format (recommended) or **SVG**
   - Choose high resolution (1920px width or higher)
   - Download the image

### Step 2: Rename Your Images

Rename your exported diagrams to match these names:

1. `agentic-ai-architecture.png` - Agentic AI Sustainability Assistant
2. `fae-metrics-architecture.png` - Enterprise FAE Metrics Platform
3. `airflow-infrastructure-architecture.png` - Hybrid Cloud Airflow Infrastructure
4. `fraud-detection-architecture.png` - Real-Time Fraud Detection Engine
5. `medallion-architecture.png` - Medallion Architecture Migration

### Step 3: Replace Placeholder Images

Copy your renamed images to this folder:
```
portfolio-vj/public/architecture/
```

This will overwrite the current placeholder images.

### Step 4: Test Locally

1. Run `npm run dev`
2. Open http://localhost:3000
3. Navigate to the Architecture section
4. Click on the diagrams to view them in full screen

---

## Adding More Architectures

Want to add additional system designs? Easy!

1. **Export your diagram** from Eraser.io
2. **Save it** in `public/architecture/` with a descriptive name (e.g., `my-new-architecture.png`)
3. **Edit** `src/data/architectureData.ts`:

```typescript
{
    id: 6,
    title: "Your Architecture Title",
    projectName: "Project/Company Name",
    description: "Detailed description of your architecture...",
    techStack: ["Tech1", "Tech2", "Tech3"],
    imagePath: "/architecture/my-new-architecture.png",
    category: "Data Engineering", // or "Machine Learning", "MLOps", "Infrastructure"
    highlights: [
        "Key highlight 1",
        "Key highlight 2",
        "Key highlight 3",
        "Key highlight 4"
    ]
}
```

4. **Save and test** - your new architecture will appear automatically!

---

## Supported Image Formats

- ✅ PNG (recommended)
- ✅ JPG/JPEG
- ✅ WebP
- ✅ SVG

## Recommended Image Specs

- **Resolution**: 1920x1080 or higher
- **Aspect Ratio**: 16:9 works best
- **File Size**: Under 2MB per image (optimize if needed)
- **Background**: Transparent or dark backgrounds work best
