from timit_raw_data import TIMITRawData
from segmentaxis import segment_axis
from pylearn2.datasets import DenseDesignMatrix
import numpy

class TIMITFrames(DenseDesignMatrix):
    def __init__(self,
                 which_set,
                 start=0,
                 stop=None,
                 start_examples=0,
                 stop_examples=None,
                 audio_only=True,
                 normalize=True,
                 speaker_filter = None,
                 phone_filter = None,
                 frame_length = 240,
                 hahn_window = None,
                 hann_window = False,
                 num_windows = 1):
#                 stft = False):
        if hahn_window!=None:
            hann_window = hahn_window
        del hahn_window
        self.__dict__.update(locals())
        del self.self
        if self.hann_window:
            assert frame_length%2==0
            self.overlap = frame_length/2 
        else:
            self.overlap = 0
            assert num_windows == 1
        self.timit_data = TIMITRawData( which_set=which_set, start=start, stop=stop, audio_only=audio_only, normalize=normalize,
                                   overlap=self.overlap, speaker_filter=speaker_filter, phone_filter=phone_filter, frame_length=frame_length + (num_windows-1)*self.overlap, stft = False )
    
        num_examples = sum( map( lambda x: len(x), self.timit_data.raw_wav ) )
        if self.stop_examples!=None:
            self.stop_examples = min( num_examples, self.stop_examples )
        else:
            self.stop_examples = num_examples
            
        
        num_examples = self.stop_examples-self.start_examples
        print "Creating ",num_examples, "examples"
        
        X = self.create_X()
        del self.timit_data
        
        if self.hann_window:
            X = self.apply_hann( X )

        super(TIMITFrames, self).__init__(X=X)# y=y)
    
    def create_X( self ):
        X = numpy.zeros( (self.stop_examples - self.start_examples, self.frame_length + (self.num_windows-1)*self.overlap), dtype='float32' )
        self.timit_data.raw_wav = list(self.timit_data.raw_wav)
        examples = 0
        for i in range(len(self.timit_data.raw_wav)):
            wav = self.timit_data.raw_wav.pop(0) # Try to save a bit of memory by removing
            for j in range(len(wav)):
                if examples>=self.stop_examples:
                    return X
                elif examples>=self.start_examples:
                    X[examples-self.start_examples,:] = wav[j]
                examples+=1
        return X
    
    def apply_hann( self, X ):
        hann = 0.5 - 0.5*numpy.cos( 2*numpy.pi*numpy.linspace(0, 1-1/float(self.frame_length), self.frame_length) )
        if self.num_windows>1:
            tiled_hann = numpy.hstack( ( numpy.zeros( (self.overlap,) ), numpy.tile( hann, numpy.ceil(self.num_windows/2.0) ), numpy.zeros( (self.overlap,) ) ) )
            hann = tiled_hann[self.overlap:] + tiled_hann[:-self.overlap]
        return X*hann
        
        
