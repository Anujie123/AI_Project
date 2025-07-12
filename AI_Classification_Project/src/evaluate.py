import joblib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from fpdf import FPDF, XPos, YPos
import os

def evaluate_models():
    data = pd.read_csv('data/processed_data.csv')
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Load models
    svm = joblib.load('models/svm_model.pkl')
    nb = joblib.load('models/nb_model.pkl')
    dnn = tf.keras.models.load_model('models/dnn_model.h5')
    
    # Evaluate
    svm_acc = accuracy_score(y, svm.predict(X))
    nb_acc = accuracy_score(y, nb.predict(X))
    dnn_acc = dnn.evaluate(X, pd.factorize(y)[0])[1]
    
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"NB Accuracy: {nb_acc:.4f}")
    print(f"DNN Accuracy: {dnn_acc:.4f}")


def generate_report():
    """Generate the final markdown report"""
    try:
        # Load processed data
        data = pd.read_csv('data/processed_data.csv')
        
        # Load model results (example - replace with your actual results)
        results = [
            {'Model': 'SVM', 'Accuracy': 0.78, 'F1': 0.75},
            {'Model': 'NB', 'Accuracy': 0.69, 'F1': 0.67},
            {'Model': 'DNN', 'Accuracy': 0.82, 'F1': 0.80}
        ]

        # Create markdown file
        with open('reports/final_report.md', 'w') as f:
            # 1. Write header
            f.write("# Wine Quality Classification Report\n\n")
            
            # 2. Dataset summary
            f.write("## Dataset Summary\n")
            f.write(f"- Total samples: {len(data)}\n")
            f.write(f"- Features: {list(data.columns)}\n")
            f.write("- Class distribution:\n")
            for cls, count in data['quality'].value_counts().items():
                f.write(f"  - {cls}: {count}\n")
            
            # 3. Results
            f.write("\n## Model Performance\n")
            f.write("| Model | Accuracy | F1-Score |\n")
            f.write("|-------|----------|----------|\n")
            for r in results:
                f.write(f"| {r['Model']} | {r['Accuracy']:.2f} | {r['F1']:.2f} |\n")
            
            # 4. Visualizations
            f.write("\n## Key Visualizations\n")
            f.write("![Confusion Matrices](figures/confusion_matrices.png)\n")
            f.write("![Training History](figures/training_history.png)\n")
            
        print("Report generated at reports/final_report.md")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        
        
def generate_pdf():
    try:
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("helvetica", size=12)

        # 1. Add title
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(0, 10, text="Wine Quality Classification Report", 
                new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(15)

        # 2. Add text content
        pdf.set_font("helvetica", size=12)
        pdf.multi_cell(0, 5, text="This report summarizes the wine quality classification project results.")
        pdf.ln(10)

        # 3. Add model performance table (hardcoded example)
        pdf.set_font("helvetica", 'B', 14)
        pdf.cell(0, 10, text="Model Performance", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        
        pdf.set_font("helvetica", size=12)
        pdf.cell(40, 10, text="Model", border=1)
        pdf.cell(40, 10, text="Accuracy", border=1)
        pdf.cell(40, 10, text="F1-Score", border=1)
        pdf.ln()
        
        models = [
            {"name": "SVM", "accuracy": "0.90", "f1": "0.88"},
            {"name": "Naïve Bayes", "accuracy": "1.00", "f1": "1.00"},
            {"name": "DNN", "accuracy": "0.80", "f1": "0.78"}
        ]
        
        for model in models:
            pdf.cell(40, 10, text=model["name"], border=1)
            pdf.cell(40, 10, text=model["accuracy"], border=1)
            pdf.cell(40, 10, text=model["f1"], border=1)
            pdf.ln()
        
        pdf.ln(15)

        # 4. Add figures if they exist
        figures_dir = "reports/figures"
        if os.path.exists(figures_dir):
            pdf.set_font("helvetica", 'B', 14)
            pdf.cell(0, 10, text="Visualizations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)
            
            for fig_name in ["confusion_matrices.png", "roc_curves.png", "training_history.png"]:
                fig_path = os.path.join(figures_dir, fig_name)
                if os.path.exists(fig_path):
                    try:
                        pdf.image(fig_path, x=10, w=180)
                        pdf.ln(5)
                        pdf.cell(0, 5, text=fig_name, align='C')
                        pdf.ln(10)
                    except:
                        pdf.multi_cell(0, 5, text=f"Could not insert {fig_name}")
                        pdf.ln(5)
        else:
            pdf.multi_cell(0, 5, text="No visualizations available")
            pdf.ln(5)

        # Save PDF
        os.makedirs("reports", exist_ok=True)
        pdf.output("reports/final_report.pdf")
        print("PDF report generated successfully at reports/final_report.pdf")
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")        


if __name__ == '__main__':
    evaluate_models()
    generate_report()
    generate_pdf()