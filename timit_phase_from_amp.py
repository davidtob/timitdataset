from timit_raw_data import TIMITRawData
from pylearn2.datasets import DenseDesignMatrix
import numpy

class TIMITPhaseFromAmp(DenseDesignMatrix):
    # A dataclass for predicting phases (as x,y of points on unit circle) from amplitudes
    
    _log_amp_mean = numpy.array([ 6.81474165,  7.26046401,  7.52874677,  7.56776545,  7.59993197,
        7.76328825,  7.74885084,  7.66979493,  7.59302478,  7.45864338,
        7.33174706,  7.2349431 ,  7.10697047,  7.00280754,  6.89841578,
        6.80349279,  6.71910556,  6.69326797,  6.6228172 ,  6.58905375,
        6.56728951,  6.55441032,  6.54265125,  6.53678243,  6.50060787,
        6.46182253,  6.42749235,  6.37465628,  6.33315936,  6.2938095 ,
        6.26050766,  6.2355035 ,  6.22121217,  6.2118505 ,  6.2077441 ,
        6.20468705,  6.1884471 ,  6.1663805 ,  6.13679662,  6.09287367,
        6.05142148,  6.01208546,  5.97639422,  5.95087402,  5.93264772,
        5.92047746,  5.91296212,  5.91611552,  5.91122373,  5.91528375,
        5.9135447 ,  5.90777627,  5.89963272,  5.88705064,  5.87238312,
        5.84800914,  5.82767709,  5.79749517,  5.77526008,  5.75057088,
        5.72303451,  5.69633816,  5.66950641,  5.64186214,  5.61747897,
        5.59685236,  5.57166932,  5.54802422,  5.52380995,  5.49827309,
        5.47270125,  5.44896232,  5.4245588 ,  5.39913341,  5.37468465,
        5.35106997,  5.3304599 ,  5.31147635,  5.29211279,  5.27075102,
        5.25593408,  5.23744124,  5.2210626 ,  5.20418941,  5.18726681,
        5.17250609,  5.1615822 ,  5.1451446 ,  5.1350982 ,  5.126375  ,
        5.11917082,  5.11584884,  5.11280432,  5.11034517,  5.1096265 ,
        5.10606246,  5.10172226,  5.09673187,  5.09033265,  5.08339316,
        5.07629378,  5.0682691 ,  5.06174902,  5.05540547,  5.05317193,
        5.04771252,  5.04446029,  5.04100303,  5.03933589,  5.03661162,
        5.03500062,  5.03534611,  5.03579187,  5.03744   ,  5.04043984,
        5.03963796,  5.04280648,  5.04170921,  5.03409247,  4.78019779], dtype='float32')
    _log_amp_std = numpy.array([ 1.66107618,  1.9114162 ,  2.10816678,  2.13505907,  2.20838948,
        2.36316446,  2.36391911,  2.39083517,  2.34130665,  2.32791461,
        2.26342592,  2.19640466,  2.13237221,  2.10202952,  2.08062851,
        2.06888003,  2.05165903,  2.01016204,  2.00357852,  1.98782965,
        1.98981139,  1.9815426 ,  1.9793071 ,  1.97229092,  1.96883885,
        1.96123938,  1.93595214,  1.93151845,  1.9036209 ,  1.89619614,
        1.87802061,  1.86477749,  1.85761318,  1.856761  ,  1.86229452,
        1.86638957,  1.8704817 ,  1.86893863,  1.85927842,  1.8764847 ,
        1.83188549,  1.8183235 ,  1.81023414,  1.8039362 ,  1.80346528,
        1.80334684,  1.81215167,  1.81358027,  1.82735601,  1.82843265,
        1.83255526,  1.83283355,  1.82966516,  1.82436397,  1.81392391,
        1.80792029,  1.79100197,  1.7883729 ,  1.77465679,  1.76543186,
        1.75694683,  1.7485576 ,  1.73886289,  1.73245423,  1.72713156,
        1.71824294,  1.71497469,  1.70693809,  1.70071897,  1.69334513,
        1.68959875,  1.6840164 ,  1.67950143,  1.67560053,  1.67185343,
        1.66906343,  1.66614831,  1.6638561 ,  1.66185733,  1.68497178,
        1.65298679,  1.65100316,  1.64491552,  1.63914021,  1.63320743,
        1.62752009,  1.6213819 ,  1.61820367,  1.61564305,  1.60978234,
        1.60676337,  1.60300009,  1.60090747,  1.59895853,  1.595088  ,
        1.59312711,  1.58947144,  1.58680537,  1.58170165,  1.57841414,
        1.57665592,  1.57346179,  1.57130631,  1.57177721,  1.56890612,
        1.57144609,  1.5697345 ,  1.56934131,  1.56907873,  1.56932531,
        1.56798753,  1.56903078,  1.568924  ,  1.56994934,  1.56800891,
        1.57014662,  1.57013387,  1.5707602 ,  1.57267556,  1.77178891], dtype='float32')
    
    def __init__( self,
                  which_set,
                  start=0,
                  stop=None,
                  log_amp = True,
                  normalize_log_amp = True,
                  speaker_filter = None,
                  phone_filter = None,
                  frame_length = 241):
        self.__dict__.update(locals())
        del self.self
        timit_data = TIMITRawData( which_set, start, stop, audio_only=True, speaker_filter=speaker_filter, phone_filter=phone_filter, stft=True, frame_length = frame_length )
        
        assert frame_length%2==1 # Want odd window so that highest frequency fourier coefficient is not constrained to be real
        
        num_examples = sum( map( lambda x: len(x), timit_data.raw_wav ) )
        self.fourier_rep_frame_length = len(timit_data.raw_wav[0][0])
        X = numpy.zeros( (num_examples, self.fourier_rep_frame_length - 1), dtype='float32' ) # Do not include first coefficient        
        y = numpy.zeros( (num_examples, 2*(self.fourier_rep_frame_length - 1) ), dtype='float32' ) # Do not include first coefficient
        
        timit_data.raw_wav = list(timit_data.raw_wav)
        examples_added = 0
        for i in reversed(range(len(timit_data.raw_wav))):
            wav = timit_data.raw_wav.pop() # Try to save a bit of memory by removing
            for j in range(len(wav)):
                X[examples_added,:] = numpy.abs( wav[j][1:] ) # amplitudes
                normalized = wav[j][1:]/X[examples_added,:]
                y[examples_added,:self.fourier_rep_frame_length-1] = numpy.real(normalized)
                y[examples_added,self.fourier_rep_frame_length-1:] = numpy.imag(normalized)
                examples_added+=1
        del timit_data
        
        if log_amp==True:
            y = y[ numpy.min(X,1)>0,:]
            X = X[ numpy.min(X,1)>0, :]
            X = numpy.log(X)
            if normalize_log_amp==True:
                assert frame_length == 241 # Hardcoded normalization currently only for this framelength
                X = (X-self._log_amp_mean)/self._log_amp_std
        
        super(TIMITPhaseFromAmp, self).__init__(X=X, y=y)
                
    def to_audio( self, amplitudes, phases ):
        assert amplitudes.shape[0]==phases.shape[0]
        rawamplitudes = numpy.exp( amplitudes*self._log_amp_std + self._log_amp_mean )
        realphases = phases[:,:self.fourier_rep_frame_length-1]
        imphases = phases[:,self.fourier_rep_frame_length-1:]
        coeffs = numpy.hstack( (numpy.zeros( (amplitudes.shape[0],1) ),rawamplitudes*(realphases + 1j*imphases) ) )
        wav = numpy.fft.irfft( coeffs, self.frame_length )
        wav = wav.reshape( (amplitudes.shape[0]*self.frame_length, 1) )
        return wav.astype('int16')
        
