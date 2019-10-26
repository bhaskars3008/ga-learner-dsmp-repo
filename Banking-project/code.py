# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Code starts here
data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()
plt.bar(loan_status.index, loan_status)
plt.show()  


# --------------
#Code starts here
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
property_and_loan = data.groupby(['Property_Area','Loan_Status'])
property_and_loan = property_and_loan.size().unstack()
property_and_loan.plot(kind = 'bar', stacked=False, figsize=(15,20))
plt.xlabel('Property Area')
plt.xticks(rotation=45)
plt.ylabel('Loan Status')
plt.show()


# --------------
#Code starts here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
education_and_loan = data.groupby(['Education', 'Loan_Status'])
education_and_loan = education_and_loan.size().unstack()
education_and_loan.plot(kind='bar', stacked=True, figsize=(20,20))
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()


# --------------
#Code starts here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
graduate = data[data['Education'] == 'Graduate']
not_graduate=data[data['Education'] == 'Not Graduate']
graduate['LoanAmount'].plot(kind='density', label='Graduate', figsize=(20,20))
not_graduate['LoanAmount'].plot(kind='density', label='Not Graduate',figsize=(20,20))

plt.legend()










#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(20,20))
ax_1.scatter(data['ApplicantIncome'],data['LoanAmount'])
ax_1.set(title = 'Applicant Income')
ax_2.scatter(data['CoapplicantIncome'], data['LoanAmount'])
ax_2.set(title = 'Coapplicant Income')
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
ax_3.scatter(data['TotalIncome'], data['LoanAmount'])
ax_3.set(title= 'Total Income')



