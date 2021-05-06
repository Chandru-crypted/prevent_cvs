# from scipy.signal import lfilter
from matplotlib import pyplot
from scipy.signal import kaiserord, lfilter, firwin, freqz
import copy
from numpy import pi, absolute

sample_rate = 30 

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
cutoff_hz = 1.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# taps = [374,
#   -28,
#   -1246,
#   -2036,
#   -1166,
#   319,
#   474,
#   -413,
#   -458,
#   382,
#   401,
#   -427,
#   -380,
#   492,
#   370,
#   -579,
#   -366,
#   696,
#   365,
#   -856,
#   -366,
#   1082,
#   369,
#   -1429,
#   -372,
#   2041,
#   374,
#   -3450,
#   -375,
#   10421,
#   16759,
#   10421,
#   -375,
#   -3450,
#   374,
#   2041,
#   -372,
#   -1429,
#   369,
#   1082,
#   -366,
#   -856,
#   365,
#   696,
#   -366,
#   -579,
#   370,
#   492,
#   -380,
#   -427,
#   401,
#   382,
#   -458,
#   -413,
#   474,
#   319,
#   -1166,
#   -2036,
#   -1246,
#   -28,
#   374]

# taps = [-0.019908809623895997, -0.06488886069297811,
#   -0.05390612002850308,
#   0.02018820608420117,
#   0.020821688342556977,
#   -0.026549380032707095,
#   0.005391110965727233,
#   0.017852537072138686,
#   -0.023357160726180125,
#   0.0074151005965933225,
#   0.016808775005335407,
#   -0.027958595035905497,
#   0.013948693122340466,
#   0.01701234318437705,
#   -0.03847615187879775,
#   0.0265920433274523,
#   0.017603101419259682,
#   -0.06096878702927477,
#   0.057437308371366395,
#   0.01811492916996912,
#   -0.14570540118324005,
#   0.26585794307082317,
#   0.684989094472488,
#   0.26585794307082317,
#   -0.14570540118324005,
#   0.01811492916996912,
#   0.057437308371366395,
#   -0.06096878702927477,
#   0.017603101419259682,
#   0.0265920433274523,
#   -0.03847615187879775,
#   0.01701234318437705,
#   0.013948693122340466,
#   -0.027958595035905497,
#   0.016808775005335407,
#   0.0074151005965933225,
#   -0.023357160726180125,
#   0.017852537072138686,
#   0.005391110965727233,
#   -0.026549380032707095,
#   0.020821688342556977,
#   0.02018820608420117,
#   -0.05390612002850308,
#   -0.06488886069297811,
#   -0.019908809623895997]

print(len(taps))

w, h = freqz(taps, worN=8000)
pyplot.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
pyplot.show()

pyplot.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
pyplot.xlim(0, cutoff_hz)
#pyplot.ylim(0.9985, 1.001)
pyplot.grid(True)
pyplot.show()

pyplot.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
pyplot.xlim(cutoff_hz + 3, 20.0)
#pyplot.ylim(0.0, 0.0025)
pyplot.grid(True)
pyplot.show()
# xlabel('Frequency (Hz)')
# ylabel('Gain')
# title('Frequency Response')
# ylim(-0.05, 1.05)
# grid(True)


#x = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,1 , 1,0, 0,  1, 1, 1, 0, 1,0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,1 , 1,0, 0,  1, 1, 1, 0, 1,0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,0, 0, 1 , 1, 1,1] 
#x = [1, 1, 0, 1, 1, 1 , 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
#x = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]

#x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
x = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w, h = freqz(x, worN=8000)
pyplot.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
pyplot.show()

pyplot.rcParams['figure.figsize'] = [((16/25) * 24), ((9/25)*24)]
pyplot.plot(x)
pyplot.savefig('1.png')
pyplot.show()
print(len(x))

y = lfilter(taps, 1.0, x)
pyplot.rcParams['figure.figsize'] = [((16/25) * 24), ((9/25)*24)]
pyplot.plot(y)
pyplot.savefig('2.png')
pyplot.show()
for i in range(len(y)):
	if (y[i] <= 0.5):
		y[i] = 0.0
	else:
		y[i] = 1.0

pyplot.rcParams['figure.figsize'] = [((16/25) * 24), ((9/25)*24)]
pyplot.plot(y)
pyplot.savefig('2.png')
pyplot.show()

print(y)

for n in range(len(y)):
	if y[n]==1.0:
		i = copy.deepcopy(n)
		while ((i + 1 < len(y))):
			if (y[i+1] == 1.0): 
				y[i+1]=0.0
				i+=1
			else:
				break

#scala gli 1.0 di 5 frame per posizionarlo alla chiusura circa
#y = [0.0,0.0,0.0,0.0,0.0] + y[0 :len(y) - 5]


print(y)
print(len(y))
pyplot.rcParams['figure.figsize'] = [((16/25) * 24), ((9/25)*24)]
pyplot.plot(y)
pyplot.savefig('3.png')
pyplot.show()



for i in range(len(y)):
	print(i ,x[i], y[i], end = ' ')
	if y[i] == 1.0 :
		print( "---- 1 here ")
	print()

print("-------- The digital filter approach {} -------".format(sum(y)))


BLINK_LIST = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,1 , 1,0, 0,  1, 1, 1, 0, 1,0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,1 , 1,0, 0,  1, 1, 1, 0, 1,0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,0, 0, 1 , 1, 1,1] 
for n in range(len(BLINK_LIST)):
	#trovo il primo 1.0
	if BLINK_LIST[n]==1.0:
		i = copy.deepcopy(n)
	#correggi 1.0 isolati: se è un 1.0 singolo (o doppio) diventa 0.0 (o 0.0 0.0)
		if sum(BLINK_LIST[i:i+6])<3.0:
				BLINK_LIST[i]=0.0
		else:
			#correggi 0.0 isolati: se ci sono 0.0 singoli (o doppi) (o tripli) diventano 1.0 (o 1.0 1.0) (o 1.0 1.0 1.0)
			while (sum(BLINK_LIST[i:i+6])>=3.0):
				BLINK_LIST[i+1]=1.0
				BLINK_LIST[i+2]=1.0
				i+=1

# print(sum(BLINK_LIST))
#ora costruisco singoli 1.0 corrispondenti al blink
for n in range(len(BLINK_LIST)):
	#trovo il primo 1.0
	if BLINK_LIST[n]==1.0:
		i = copy.deepcopy(n)
		while (BLINK_LIST[i+1]==1.0):
			BLINK_LIST[i+1]=0.0
			i+=1

#scala gli 1.0 di 5 frame per posizionarlo alla chiusura circa
BLINK_LIST=[0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-5]
# print(BLINK_LIST)
print("-------- The previous approach {} -------".format(sum(BLINK_LIST)))