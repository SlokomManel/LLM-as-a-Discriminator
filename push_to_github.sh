#!/bin/bash
# push_to_github.sh
# Push all code to private GitHub repo

echo "📤 GitHub Push Script"
echo "=================================================="
echo ""
echo "This will:"
echo "  1. Stage all files (except .gitignore excludes)"
echo "  2. Create initial commit"
echo "  3. Push to GitHub"
echo ""
echo "Required: You must have created the repo on GitHub first!"
echo "         https://github.com/new"
echo ""
echo "=================================================="
echo ""

cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Check git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized. Run: git init"
    exit 1
fi

# Add all files (respecting .gitignore)
echo "🔄 Staging files..."
git add .
echo "✅ Staged $(git diff --cached --name-only | wc -l) files"
echo ""

# Show what will be committed
echo "📋 Files to commit:"
git diff --cached --name-only
echo ""

# Create commit
echo "💾 Creating commit..."
git commit -m "Initial commit: Phase 2 multi-model LLM discrimination analysis

- 1,603 trials across 6 synthetic methods (CART, GReat, TVAE, CTGAN, TabDDPM, GaussianCopula)
- Tested Google Gemini models (2.5-flash, 2.5-pro)
- Privacy rankings: CART best (0.5% discrimination), GaussianCopula worst (1.3%)
- Analysis shows excellent synthetic data privacy properties
- Ready for Phase 2 continuation after midnight UTC quota reset"

echo ""
echo "⚠️  Next steps:"
echo ""
echo "1. If you haven't created the GitHub repo yet:"
echo "   → Go to: https://github.com/new"
echo "   → Name: LLM-as-a-Discriminator"
echo "   → Select: Private"
echo "   → Click: Create repository"
echo ""
echo "2. Run these commands to finish pushing:"
echo ""
echo "   git remote add origin https://github.com/SlokomManel/LLM-as-a-Discriminator.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "   When prompted for password, use a GitHub Personal Access Token:"
echo "   → Generate at: https://github.com/settings/tokens/new"
echo "   → Scopes: repo (all)"
echo ""
