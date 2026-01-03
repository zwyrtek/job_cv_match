"""
Flask API for Job-CV Matching
Serves the trained neural network and provides predictions with skill gap analysis
NOW WITH TECHNICAL AND SOFT SKILLS CATEGORIZATION
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
from train_model_v2 import JobCVMatcherV2

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load trained model
matcher = JobCVMatcherV2()
try:
    matcher.load_model('models')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Please run train_model.py first to train the model.")

# Technical skills keywords
TECHNICAL_KEYWORDS = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
    'go', 'rust', 'scala', 'r', 'matlab', 'sql', 'html', 'css',

    # Frameworks & Libraries
    'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'fastapi',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'jquery',

    # Databases
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
    'sql server', 'dynamodb', 'firebase',

    # DevOps & Cloud
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'gitlab', 'github actions',
    'terraform', 'ansible', 'ci/cd', 'linux', 'unix', 'bash',

    # Tools & Technologies
    'git', 'jira', 'confluence', 'postman', 'swagger', 'api', 'rest', 'graphql',
    'microservices', 'kafka', 'rabbitmq', 'nginx', 'apache',

    # Data & ML
    'machine learning', 'deep learning', 'data science', 'nlp', 'computer vision',
    'data analysis', 'statistics', 'big data', 'hadoop', 'spark',

    # Testing
    'junit', 'pytest', 'selenium', 'unit testing', 'integration testing', 'tdd',

    # Architecture
    'architecture', 'design patterns', 'oop', 'functional programming', 'system design'
]

# Soft skills keywords
SOFT_SKILLS_KEYWORDS = [
    # Leadership & Management
    'leadership', 'team lead', 'management', 'mentoring', 'coaching', 'delegation',

    # Communication
    'communication', 'presentation', 'public speaking', 'writing', 'documentation',
    'stakeholder management', 'client communication',

    # Collaboration
    'teamwork', 'collaboration', 'cross-functional', 'team player', 'cooperative',

    # Problem Solving
    'problem solving', 'analytical', 'critical thinking', 'decision making', 'troubleshooting',

    # Work Style
    'agile', 'scrum', 'kanban', 'remote work', 'self-motivated', 'proactive',
    'time management', 'organization', 'multitasking', 'adaptable', 'flexible',

    # Soft Traits
    'creative', 'innovative', 'detail-oriented', 'customer-focused', 'results-driven',
    'entrepreneurial', 'strategic thinking'
]

def categorize_skill(skill):
    """Determine if skill is technical or soft skill"""
    skill_lower = skill.lower()

    # Check technical keywords
    for tech_keyword in TECHNICAL_KEYWORDS:
        if tech_keyword in skill_lower or skill_lower in tech_keyword:
            return 'technical'

    # Check soft skill keywords
    for soft_keyword in SOFT_SKILLS_KEYWORDS:
        if soft_keyword in skill_lower or skill_lower in soft_keyword:
            return 'soft'

    # Default: if contains common technical indicators
    tech_indicators = ['programming', 'development', 'framework', 'database', 'tool', 'language']
    for indicator in tech_indicators:
        if indicator in skill_lower:
            return 'technical'

    # Default to technical if unsure (most job requirements are technical)
    return 'technical'

def extract_skills_list(text):
    """Extract individual skills from text"""
    if not text or text.strip() == '':
        return []

    # Split by common delimiters
    skills = re.split(r'[,\n•\-\|]', text)
    skills = [s.strip() for s in skills if s.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower not in seen and len(skill) > 2:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills

def extract_years_experience(cv_text):
    """Try to extract years of experience from CV"""
    if not cv_text:
        return 0

    # Look for patterns like "5 years", "5+ years", "5-7 years"
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'experience\s*:?\s*(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\s*years?\s*in',
    ]

    for pattern in patterns:
        match = re.search(pattern, cv_text.lower())
        if match:
            return int(match.group(1))

    return 0

def analyze_skill_gaps(job_description, cv_text):
    """Analyze which skills from job description are missing in CV - categorized by type"""
    # Extract skills from job description
    job_skills = extract_skills_list(job_description)
    cv_skills = extract_skills_list(cv_text)

    # Convert to lowercase for comparison
    cv_skills_lower = [s.lower() for s in cv_skills]

    # Categorize matched and missing skills
    matched_technical = []
    matched_soft = []
    missing_technical = []
    missing_soft = []

    for job_skill in job_skills:
        job_skill_lower = job_skill.lower()

        # Check for exact match or partial match
        is_matched = False
        for cv_skill_lower in cv_skills_lower:
            if (job_skill_lower in cv_skill_lower or
                    cv_skill_lower in job_skill_lower or
                    job_skill_lower == cv_skill_lower):
                is_matched = True
                break

        # Categorize the skill
        skill_type = categorize_skill(job_skill)

        if is_matched:
            if skill_type == 'technical':
                matched_technical.append(job_skill)
            else:
                matched_soft.append(job_skill)
        else:
            if skill_type == 'technical':
                missing_technical.append(job_skill)
            else:
                missing_soft.append(job_skill)

    return {
        'matchedTechnical': matched_technical[:15],
        'matchedSoft': matched_soft[:10],
        'missingTechnical': missing_technical[:15],
        'missingSoft': missing_soft[:10]
    }

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Job-CV Matching API',
        'endpoints': {
            '/api/evaluate': 'POST - Evaluate job-CV match'
        }
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_match():
    """
    Evaluate match between job description and candidate CV

    Request body:
    {
        "jobDescription": "...",
        "candidateCV": "..."
    }

    Returns:
    {
        "score": 7.5,
        "scoreCategory": "Good Match",
        "matchedTechnical": [...],
        "matchedSoft": [...],
        "missingTechnical": [...],
        "missingSoft": [...],
        "explanation": "..."
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        job_description = data.get('jobDescription', '').strip()
        candidate_cv = data.get('candidateCV', '').strip()

        if not job_description or not candidate_cv:
            return jsonify({'error': 'Both jobDescription and candidateCV are required'}), 400

        # Extract years of experience from CV
        years_exp = extract_years_experience(candidate_cv)

        # Get prediction from neural network
        score = matcher.predict(
            job_description=job_description,
            cv_text=candidate_cv,
            years_experience=years_exp,
            skills=candidate_cv  # Use full CV text for skills extraction
        )

        # Round score to 1 decimal place
        score = round(float(score), 1)

        # Determine score category
        if score >= 8:
            category = "Excellent Match"
            color = "ok"
        elif score >= 6:
            category = "Good Match"
            color = "ok"
        elif score >= 4:
            category = "Potential Match"
            color = "warn"
        else:
            category = "Poor Match"
            color = "bad"

        # Analyze skill gaps with categorization
        skill_analysis = analyze_skill_gaps(job_description, candidate_cv)

        # Count skills
        matched_tech_count = len(skill_analysis['matchedTechnical'])
        matched_soft_count = len(skill_analysis['matchedSoft'])
        missing_tech_count = len(skill_analysis['missingTechnical'])
        missing_soft_count = len(skill_analysis['missingSoft'])

        total_matched = matched_tech_count + matched_soft_count
        total_missing = missing_tech_count + missing_soft_count

        # Generate explanation
        if score >= 8:
            explanation = f"The candidate demonstrates excellent alignment with the role requirements. "
            explanation += f"Strong technical foundation with {matched_tech_count} relevant technical competencies"
            if matched_soft_count > 0:
                explanation += f" and {matched_soft_count} valued soft skills"
            explanation += "."
        elif score >= 6:
            explanation = f"The candidate shows good potential for this position. "
            explanation += f"They possess {matched_tech_count} relevant technical skills"
            if matched_soft_count > 0:
                explanation += f" and {matched_soft_count} complementary soft skills"
            if missing_tech_count > 0:
                explanation += f", though {missing_tech_count} technical areas could be developed further"
            explanation += "."
        elif score >= 4:
            explanation = f"The candidate has foundational qualifications but shows notable gaps. "
            explanation += f"While they demonstrate {matched_tech_count} technical competencies"
            if missing_tech_count > 0:
                explanation += f", {missing_tech_count} key technical requirements are not evident"
            if missing_soft_count > 0:
                explanation += f" and {missing_soft_count} soft skill areas need strengthening"
            explanation += "."
        else:
            explanation = f"The candidate's profile shows limited alignment with role requirements. "
            if matched_tech_count > 0:
                explanation += f"Only {matched_tech_count} technical skills match"
            else:
                explanation += "Technical skills do not align well"
            if missing_tech_count > 0:
                explanation += f", with {missing_tech_count} critical technical competencies missing"
            explanation += "."

        if years_exp > 0:
            explanation += f" The candidate has {years_exp} years of relevant experience."

        return jsonify({
            'score': score,
            'scoreCategory': category,
            'scoreColor': color,
            'yearsExperience': years_exp,
            'matchedTechnical': skill_analysis['matchedTechnical'],
            'matchedSoft': skill_analysis['matchedSoft'],
            'missingTechnical': skill_analysis['missingTechnical'],
            'missingSoft': skill_analysis['missingSoft'],
            'explanation': explanation
        })

    except Exception as e:
        print(f"Error in evaluate_match: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': matcher.model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("JOB-CV MATCHING API SERVER")
    print("="*60)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /              - API info")
    print("  POST /api/evaluate  - Evaluate job-CV match")
    print("  GET  /api/health    - Health check")
    print("\n" + "="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
