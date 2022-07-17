from flask import Flask,render_template,request
import pickle

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

app=Flask(__name__)
@app.route('/',methods=["POST","GET"])
def result():
    if (request.method == "POST"):
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        pain = int(mydict['pain'])
        runnynose = int(mydict['runnynose'])
        diffbreath = int(mydict['diffbreath'])

        inputfeatures = [fever, pain, age, runnynose, diffbreath]
        corona = clf.predict_proba([inputfeatures])[0][1]
        print(corona)

        return render_template('show.html', inf=round(corona * 100))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True,port=3456)











