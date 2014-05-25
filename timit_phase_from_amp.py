from timit_raw_data import TIMITRawData
from pylearn2.datasets import DenseDesignMatrix


class TIMITPhaseFromAmp(DenseDesignMatrix):
    # A dataclass for predicting phases (as x,y of points on unit circle) from amplitudes
    def __init__( self,
                  which_set,
                  start=0,
                  stop=None,
                  speaker_filter = None,
                  phone_filter = None,
                  frame_length = 241):
        self.__dict__.update(locals())
        del self.self
        timit_data = TIMITRawData( which_set, start, stop, audio_only=True, speaker_filter=speaker_filter, phone_filter=phone_filter, stft=True )
        
        assert frame_length%2==1 # Want odd window so that highest frequency fourier coefficient is not constrained to be real
        
        num_examples = sum( map( lambda x: len(x), timit_data.raw_wav ) )
        fourier_rep_frame_length = len(timit_data.raw_wav[0][0])
        X = numpy.zeros( (num_examples, fourier_rep_frame_length - 1), dtype='float32' ) # Do not include first coefficient
        y = numpy.zeros( (num_examples, 2*(fourier_rep_frame_length - 1) ), dtype='float32' ) # Do not include first coefficient
        
        timit_data.raw_wav = list(timit_data.raw_wav)
        examples_added = 0
        for i in reversed(range(len(timit_data.raw_wav))):
            wav = timit_data.raw_wav.pop() # Try to save a bit of memory by removing
            for j in range(len(wav)):
                X[examples_added,:] = numpy.abs( wav[j][1:] ) # amplitudes
                normalized = wav[j][1:]/X[examples_added,:]
                y[examples_added,:fourier_rep_frame_length-1] = numpy.real(normalized)
                y[examples_added,fourier_rep_frame_length-1:] = numpy.imag(normalized)
                examples_added+=1
        del timit_data
        
        super(TIMITPhaseFromAmp, self).__init__(X=X, y=y)
