"""
Pylearn2 wrapper for the TIMIT dataset
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "dumouliv@iro"


import os.path
import functools
import numpy
from pylearn2.utils.iteration import resolve_iterator_class, RandomUniformSubsetIterator
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
#from research.code.pylearn2.space import (
#    VectorSequenceSpace,
#    IndexSequenceSpace
#)
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from segmentaxis import segment_axis
from iteration import FiniteDatasetIterator
from pylearn2.datasets import DenseDesignMatrix
import scipy.stats

class TIMITRawData(object):
    def __init__(self,
                 which_set,
                 start=0,
                 stop=None,
                 audio_only=False,
                 speaker_filter = None,
                 phone_filter = None,
                 frame_length = 240,
                 stft = False):
        self.__dict__.update(locals())
        del self.self

        ## Load data from disk
        #if which_set=='train_train' or which_set=='train_valid':
        #    self._load_data('train') # In this case we further split the training from disk into a training set and a validation set
        #else:
        # Load data into memory
        self._load_data(which_set)
        # Process data
        self.slice_data() 
        if self.speaker_filter!=None:
            self.filter_speakers()
        if self.phone_filter!=None:
            self.filter_phones()
        if self.stft==True:
            self.filter_based_on_frame_length()
            self.compute_stft()
    
    def _load_data(self, which_set):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Create file paths
        timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       "timit/readable")
        speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        phonemes_list_path = os.path.join(timit_base_path,
                                          "reduced_phonemes.pkl")
        words_list_path = os.path.join(timit_base_path, "words.pkl")
        speaker_features_list_path = os.path.join(timit_base_path,
                                                  "spkr_feature_names.pkl")
        speaker_id_list_path = os.path.join(timit_base_path,
                                            "speakers_ids.pkl")
        raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        #phonemes_path = os.path.join(timit_base_path,
        #                             which_set + "_x_phonemes.npy")
        phone_nums_path   = os.path.join(timit_base_path,
                                     which_set + "_x_compact_phone_nums.npy")
        phone_offsets_path = os.path.join(timit_base_path,
                                     which_set + "_x_compact_phone_offsets.npy")
        #words_path = os.path.join(timit_base_path, which_set + "_x_words.npy")
        speaker_path = os.path.join(timit_base_path,
                                    which_set + "_spkr.npy")

        # Load data. For now most of it is not used, as only the acoustic
        # samples are provided, but this is bound to change eventually.
        # Global data
        if not self.audio_only:
            self.speaker_info_list = serial.load(
                speaker_info_list_path
            ).tolist().toarray()
            self.speaker_id_list = serial.load(speaker_id_list_path)
            self.speaker_features_list = serial.load(speaker_features_list_path)
            self.words_list = serial.load(words_list_path)
            self.phonemes_list = serial.load(phonemes_list_path)
        # Set-related data
        self.raw_wav = serial.load(raw_wav_path)        
        if not self.audio_only:
            self.phone_nums    = serial.load(phone_nums_path)
            self.phone_offsets = serial.load(phone_offsets_path)
            #self.phonemes = serial.load(phonemes_path)
            #self.phones = serial.load(phones_path)
            #self.words = serial.load(words_path)
            self.speaker_id = numpy.asarray(serial.load(speaker_path), 'int')
            
    def slice_data( self ):
        if self.stop is None:
            self.stop = len(self.raw_wav)
        self.raw_wav = self.raw_wav[self.start:self.stop]
        if not self.audio_only:
            self.phone_nums    = self.phone_nums[self.start:self.stop]
            self.phone_offsets = self.phone_offsets[self.start:self.stop]
            #self.phonemes = self.phonemes[start:stop]
            #self.words = self.words[start:stop]
    
    def filter_speakers( self ): # keep only utterances by some speakers
        if self.speaker_filter != None:
            keep_indices = []
            for i,sid in enumerate(self.speaker_id):
                if sid in self.speaker_filter:
                    keep_indices.append(i)
            self.raw_wav       = self.raw_wav[keep_indices]
            self.phone_nums    = self.phone_nums[keep_indices]
            self.phone_offsets = self.phone_offsets[keep_indices]
            self.speaker_id    = self.speaker_id[keep_indices]
    
    def filter_phones( self ): # Filter out phones that we do not want to include (making a new sequence for each phone we do include)
        if self.phone_filter != None :
            new_raw_wav = []
            new_phone_nums = []
            new_phone_offsets = []
            new_speaker_id = []
            for sequence_id, phn_nums in enumerate(self.phone_nums):
                for phn_idx, phn_num in enumerate( phn_nums ):
                    if phn_num in self.phone_filter:
                        phn_start = self.phone_offsets[sequence_id][phn_idx]
                        phn_end   = (list(self.phone_offsets[sequence_id]) + [len(self.raw_wav[sequence_id])-1])[phn_idx+1]
                        if self.mid_third == True:
                            phn_start, phn_end = (phn_start + (phn_end-phn_start)/4, phn_end - (phn_end - phn_start)/4)
                        if phn_start+self.frames_per_example<phn_end:
                            new_raw_wav.append   (  self.raw_wav[sequence_id][phn_start:phn_end]  )
                            new_phone_nums.append(  numpy.array([phn_num]) )
                            new_phone_offsets.append( numpy.array([0]) )
                            new_speaker_id.append( self.speaker_id[sequence_id] )
            self.raw_wav = new_raw_wav
            self.phone_nums = new_phone_nums
            self.phone_offsets = new_phone_offsets
            self.speaker_id = new_speaker_id
    
    def filter_based_on_frame_length( self ): # Filter out all utterances that are shorter than frame length
        idcs_to_delete = []
        for i,utterance in enumerate(self.raw_wav):
            if len(utterance)<self.frame_length:
                idcs_to_delete.append( i )
        for i in reversed( idcs_to_delete ):
            del self.raw_wav[i]
            del self.phone_nums[i]
            del self.phone_offsets[i]
            del self.speaker_id[i]

    def compute_stft( self ):
        # Replace each utterance with its stft with window length self.frame_length
        print "Computing STFT"
        delete_idcs = []
        for i,utterance in enumerate(self.raw_wav):
            frames = segment_axis( self.raw_wav[i], length=self.frame_length, overlap=0 )
            self.raw_wav[i] = numpy.fft.rfft( frames )
        print "Done"
