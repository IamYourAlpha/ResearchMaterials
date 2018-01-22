import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt



accepted = ctrl.Antecedent( np.arange(0,22990,1), 'accepted')
wrongAnswer = ctrl.Antecedent( np.arange(0, 14537, 1), 'wrongAnswer')
timeLimitExceed = ctrl.Antecedent( np.arange(0, 8209, 1), 'timeLimitExceed')
submission = ctrl.Antecedent( np.arange(0,40493, 1), 'submission')
solved = ctrl.Antecedent( np.arange(0,14794,1), 'solved')


difficulty = ctrl.Consequent( np.arange(0,3,1), 'difficulty')

accepted['low'] = fuzz.trimf(accepted.universe, [0,0,645])
accepted['medium'] = fuzz.trimf(accepted.universe, [32,1290,2568])
accepted['high'] = fuzz.trimf(accepted.universe, [900,3000,6340])
accepted['veryhigh'] = fuzz.trimf(accepted.universe, [5000,8500,15000])
accepted['advance'] = fuzz.trimf(accepted.universe, [11960,22900,22900])

wrongAnswer['low'] = fuzz.trimf(wrongAnswer.universe, [0,0,802])
wrongAnswer['medium'] = fuzz.trimf(wrongAnswer.universe, [590,2290,6178])
wrongAnswer['High'] = fuzz.trimf(wrongAnswer.universe, [3000,6500,11959])
wrongAnswer['veryHigh'] = fuzz.trimf(wrongAnswer.universe, [8000,14537,14537])


timeLimitExceed['low'] = fuzz.trimf(timeLimitExceed.universe, [0,0,500])
timeLimitExceed['medium'] = fuzz.trimf(timeLimitExceed.universe, [320, 500, 1189])
timeLimitExceed['High'] = fuzz.trimf(timeLimitExceed.universe, [1000, 3000, 8209])
timeLimitExceed['veryHigh'] = fuzz.trimf(timeLimitExceed.universe, [3000, 8209, 8209])


submission['low'] = fuzz.trimf(submission.universe, [0,0,1196])
submission['medium'] = fuzz.trimf(submission.universe, [2910, 4405, 10239])
submission['High'] = fuzz.trimf(submission.universe, [8350, 10239, 23955])
submission['veryHigh'] = fuzz.trimf(submission.universe, [17257, 40493, 40493])

solved['low'] = fuzz.trimf(solved.universe, [0,0,545])
solved['medium'] = fuzz.trimf(solved.universe, [500,1702,3536])
solved['high'] = fuzz.trimf(solved.universe, [1702,3536,7371])
solved['veryHigh'] = fuzz.trimf(solved.universe, [7371,14794,14794])


difficulty.automf(3)

accepted.view()
wrongAnswer.view()
timeLimitExceed.view()
submission.view()
solved.view()
rule1 = ctrl.Rule(accepted['high'] & wrongAnswer['low'] & timeLimitExceed['low'], difficulty['poor'])
rule2 = ctrl.Rule(accepted['low'] & wrongAnswer['High'] & timeLimitExceed['High'], difficulty['good'])

rule_add = ctrl.ControlSystem([rule1, rule2])
system  = ctrl.ControlSystemSimulation(rule_add)

system.input['accepted'] = 3500
system.input['wrongAnswer'] = 0
system.input['timeLimitExceed'] = 0
system.compute()
print system.output['difficulty']
'''
x_ac = np.arange(0, 9409, 1)
x_wa = np.arange(0, 14537, 1)
x_dif = np.arange(0,3,1)

accAdv = fuzz.trimf(x_ac, [0,0,645])
accVhard = fuzz.trimf(x_ac, [32,1284,2568])
accHard = fuzz.trimf(x_ac, [823,3000,6340])
acMed  = fuzz.trimf(x_ac, [2283,5846, 9409])



dif_lo = fuzz.trimf(x_dif, [0,0,1])
dif_med = fuzz.trimf(x_dif, [0,1,2])
dif_hi = fuzz.trimf(x_dif, [1,2,2])

fig, (ax, ax2) = plt.subplots(nrows=2)
ax.plot(x_dif, dif_lo, 'b', linewidth=1.5, label='Easy')
ax.plot(x_dif, dif_med, 'g', linewidth=1.5, label='Medium')
ax.plot(x_dif, dif_hi, 'r', linewidth=1.5, label='Hard')
ax.set_xlabel('Difficulty class')
ax.set_ylabel('Degree of membership')
ax.legend()
ax.set_title('Problem Difficulty')

ax2.plot(x_ac, accAdv)
ax2.plot(x_ac, accVhard)
ax2.plot(x_ac, accHard)
ax2.plot(x_ac, acMed)
plt.show()
'''





