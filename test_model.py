"""
Test Script for Job-CV Matching System
Tests the trained model with sample data
"""

from train_model import JobCVMatcher

def test_model():
    print("="*60)
    print("TESTING JOB-CV MATCHING MODEL")
    print("="*60)
    
    # Load model
    print("\n1. Loading trained model...")
    matcher = JobCVMatcher()
    try:
        matcher.load_model('models')
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        print("   Please run train_model.py first.")
        return
    
    # Test Case 1: Good match
    print("\n2. Testing Case 1: Good Match")
    print("-" * 60)
    
    job_desc_1 = """
    Software Engineer Position
    
    We are looking for a Python developer with experience in machine learning
    and web development. Required skills include:
    - Python programming
    - TensorFlow or PyTorch
    - Flask or Django
    - REST API development
    - 3+ years experience
    """
    
    cv_1 = """
    John Doe
    Software Engineer
    
    Experience: 5 years
    
    Skills:
    - Python (expert level)
    - TensorFlow and Keras
    - Flask web framework
    - REST API design
    - Docker and AWS
    
    Experience:
    Senior Python Developer at Tech Corp (3 years)
    - Built ML models using TensorFlow
    - Developed REST APIs with Flask
    - Deployed applications on AWS
    """
    
    score_1 = matcher.predict(job_desc_1, cv_1, years_experience=5, skills=cv_1)
    print(f"   Job: Software Engineer (Python, ML, Flask)")
    print(f"   Candidate: 5 years exp, Python expert, TensorFlow, Flask")
    print(f"   Score: {score_1:.1f}/10")
    
    # Test Case 2: Poor match
    print("\n3. Testing Case 2: Poor Match")
    print("-" * 60)
    
    job_desc_2 = """
    Senior Java Backend Developer
    
    Requirements:
    - 8+ years Java development
    - Spring Boot framework
    - Microservices architecture
    - PostgreSQL database
    - Kubernetes deployment
    """
    
    cv_2 = """
    Jane Smith
    Frontend Developer
    
    Experience: 2 years
    
    Skills:
    - React and JavaScript
    - HTML/CSS
    - Basic TypeScript
    - Git version control
    
    Experience:
    Junior Frontend Developer (2 years)
    - Built responsive websites with React
    - Collaborated with design team
    """
    
    score_2 = matcher.predict(job_desc_2, cv_2, years_experience=2, skills=cv_2)
    print(f"   Job: Senior Java Backend Developer (8+ years)")
    print(f"   Candidate: 2 years exp, Frontend only, React/JavaScript")
    print(f"   Score: {score_2:.1f}/10")
    
    # Test Case 3: Medium match
    print("\n4. Testing Case 3: Moderate Match")
    print("-" * 60)
    
    job_desc_3 = """
    Data Scientist
    
    Requirements:
    - Python programming
    - Machine Learning experience
    - Statistical analysis
    - SQL databases
    - Data visualization
    - 3-5 years experience
    """
    
    cv_3 = """
    Alex Johnson
    Data Analyst
    
    Experience: 3 years
    
    Skills:
    - Python and R
    - SQL (PostgreSQL, MySQL)
    - Tableau and PowerBI
    - Statistical modeling
    - Excel advanced
    
    Experience:
    Data Analyst at Finance Corp (3 years)
    - Created dashboards and reports
    - Analyzed customer data
    - Built predictive models in Python
    """
    
    score_3 = matcher.predict(job_desc_3, cv_3, years_experience=3, skills=cv_3)
    print(f"   Job: Data Scientist (Python, ML, SQL, 3-5 years)")
    print(f"   Candidate: 3 years Data Analyst, Python, SQL, some ML")
    print(f"   Score: {score_3:.1f}/10")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Case 1 (Good Match):     {score_1:.1f}/10 {'✓' if score_1 >= 6 else '✗'}")
    print(f"Case 2 (Poor Match):     {score_2:.1f}/10 {'✓' if score_2 < 5 else '✗'}")
    print(f"Case 3 (Moderate Match): {score_3:.1f}/10 {'✓' if 4 <= score_3 < 8 else '✗'}")
    print("\nModel is working correctly!" if (score_1 >= 6 and score_2 < 5 and 4 <= score_3 < 8) else "\nModel predictions may need review.")
    print("="*60)

if __name__ == "__main__":
    test_model()
