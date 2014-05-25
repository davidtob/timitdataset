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
from research.code.pylearn2.space import (
    VectorSequenceSpace,
    IndexSequenceSpace
)
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from research.code.scripts.segmentaxis import segment_axis
from research.code.pylearn2.utils.iteration import FiniteDatasetIterator
import scipy.stats

class TIMITOnTheFly(Dataset):
    """
    Frame-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)

    # Mean and standard deviation of the acoustic samples from the whole
    # dataset (train, valid, test).
    _mean = 0.0035805809921434142
    _std = 542.48824133746177

    def __init__(self, which_set, frame_length,
                 overlap=0,
                 frames_per_example=1,
                 output_frames_per_example=1,
                 start=0, stop=None, audio_only=False,
                 rng=_default_seed,
                 noise = False,
                 noise_decay = False,
                 speaker_filter = None,
                 phone_filter = None,
                 mid_third = False):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
        frames_per_example : int, optional
            Number of frames in a training example. Defaults to 1.
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        audio_only : bool, optional
            Whether to load only the raw audio and no auxiliary information.
            Defaults to `False`.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        assert frame_length==1 # Longer frame length implemented through output_frames_per_example
        self.frame_length = 1#frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example
        self.output_frames_per_example = output_frames_per_example
        self.offset = self.frame_length - self.overlap
        self.audio_only = audio_only
        self.noise = noise
        self.noise_decay = noise_decay
        self.speaker_filter = speaker_filter
        self.phone_filter = phone_filter
        self.mid_third = mid_third
        self.use_examples = None

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        # Load data from disk
        if which_set=='train_train' or which_set=='train_valid':
            self._load_data('train') # In this case we further split the training from disk into a training set and a validation set
        else:
            self._load_data(which_set)
        
        # Slice data
        if stop is None:
            stop = len(self.raw_wav)
        self.raw_wav = self.raw_wav[start:stop]
        if not self.audio_only:
            self.phone_nums    = self.phone_nums[start:stop]
            self.phone_offsets = self.phone_offsets[start:stop]
            #self.phonemes = self.phonemes[start:stop]
            #self.words = self.words[start:stop]
        
        # filter based on speaker
        if self.speaker_filter != None:
            keep_indices = []
            for i,sid in enumerate(self.speaker_id):
                if sid in self.speaker_filter:
                    keep_indices.append(i)
            self.raw_wav       = self.raw_wav[keep_indices]
            self.phone_nums    = self.phone_nums[keep_indices]
            self.phone_offsets = self.phone_offsets[keep_indices]
            self.speaker_id    = self.speaker_id[keep_indices]

         # Filter out phones that we do not want to include (making a new sequence for each phone we do include)
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
        
        examples_per_sequence = [0] + map( lambda x: len(x) - self.frames_per_example - self.output_frames_per_example + 1, self.raw_wav )
        
        self.cumulative_example_indexes = numpy.cumsum(examples_per_sequence)        
        self.num_examples = self.cumulative_example_indexes[-1]

        # If requested, make further split of disk training set (only works well if the number of examples is small)
        if which_set =='train_train' or which_set=='train_valid':
            digit = numpy.digitize(range(self.num_examples), self.cumulative_example_indexes) - 1
            ex_indices = zip(digit, numpy.array(range(self.num_examples)) - self.cumulative_example_indexes[digit])
            numpy.random.shuffle( ex_indices )
            if which_set == 'train_train':
                self.use_indices = ex_indices[:int(self.num_examples*0.8)]
            elif which_set=='train_valid':
                self.use_indices = ex_indices[int(self.num_examples*0.8):]
            self.num_examples = len(self.use_indices)

        print "number of examples", self.num_examples
            
        self.samples_sequences = self.raw_wav

        # DataSpecs
        features_space = VectorSpace(
            dim=self.frame_length * self.frames_per_example
        )
        features_source = 'features'
        def features_map_fn(indices, batch_buffer):            
            for i, (sequence_index, example_index) in enumerate(self._fetch_index(indices)):
                batch_buffer[i,:] = self.samples_sequences[sequence_index][example_index:example_index
                                                                           + self.frames_per_example].ravel()
            batch_buffer[:,:] = (batch_buffer - TIMITOnTheFly._mean) / TIMITOnTheFly._std # Modify in place
        
        if self.noise_decay==False:
            self.noiseprofile = numpy.ones( (1, self.frames_per_example) )
        else:
            self.noiseprofile = numpy.linspace( 1, 0, self.frames_per_example ).reshape( (1,self.frames_per_example ) )
        
        def features_map_fn_noise(indices, batch_buffer):
            features_map_fn(indices, batch_buffer )
            if isinstance(self.noise,float):
                batch_buffer[:,:] = batch_buffer + numpy.random.normal( 0, self.noise, batch_buffer.shape )*self.noiseprofile # Modify in place
            elif isinstance(self.noise,list):
                #noises = numpy.random.choice( self.noise, (batch_buffer.shape[0], 1) ) LisaLab does not have numpy 1.7.0 yet
                noises = numpy.array(self.noise).reshape( (len(self.noise), 1) )[ numpy.random.randint( 0, len(self.noise), batch_buffer.shape[0] ) ]
                batch_buffer[:,:] = batch_buffer + numpy.random.normal( 0, 1, batch_buffer.shape )*noises*self.noiseprofile # Modify in place
                    
        targets_space = VectorSpace(dim=self.frame_length*self.output_frames_per_example)
        targets_source = 'targets'
        def targets_map_fn(indices, batch_buffer):
            for i, (sequence_index, example_index) in enumerate(self._fetch_index(indices)):
                batch_buffer[i,:] = self.samples_sequences[sequence_index][example_index + self.frames_per_example
                                                                           :example_index + self.frames_per_example + self.output_frames_per_example].ravel()
            batch_buffer[:,:] = (batch_buffer - TIMITOnTheFly._mean) / TIMITOnTheFly._std # Modify in place

        space_components = [features_space, targets_space]
        source_components = [features_source, targets_source]
        if self.noise == False:
            map_fn_components = [features_map_fn, targets_map_fn]
        else:
            map_fn_components = [features_map_fn_noise, targets_map_fn]
        batch_components = [None, None]

        if not self.audio_only:
            num_phones = 62
            phones_space = IndexSpace(max_labels=num_phones, dim=1,
                                      dtype=str(self.phone_nums[0].dtype))
            phones_source = 'phones'
            def phones_map_fn(indices, batch_buffer):
                for i, (sequence_index, example_index) in enumerate(self._fetch_index(indices)):
                    digit = numpy.digitize([example_index + self.frames_per_example], self.phone_offsets[sequence_index])[0] - 1
                    batch_buffer[i,0] =  self.phone_nums[sequence_index][digit]

#            num_phonemes = numpy.max([numpy.max(sequence) for sequence
                                      #in self.phonemes]) + 1
#            phonemes_space = IndexSpace(max_labels=num_phonemes, dim=1,
#                                        dtype=str(self.phonemes_sequences[0].dtype))
#            phonemes_source = 'phonemes'
#            def phonemes_map_fn(indexes):
#                rval = []
#                for sequence_index, example_index in self._fetch_index(indexes):
#                    rval.append(self.phonemes_sequences[sequence_index][example_index
#                        + self.frames_per_example].ravel())
#                return rval

#            num_words = numpy.max([numpy.max(sequence) for sequence
#                                   in self.words]) + 1
#            words_space = IndexSpace(max_labels=num_words, dim=1,
                                     #dtype=str(self.words_sequences[0].dtype))
#            words_source = 'words'
#            def words_map_fn(indexes):
#                rval = []
#                for sequence_index, example_index in self._fetch_index(indexes):
#                    rval.append(self.words_sequences[sequence_index][example_index
#                        + self.frames_per_example].ravel())
#                return rval

            space_components.extend([phones_space])#, phonemes_space,
                                     #words_space])
            source_components.extend([phones_source])#, phonemes_source,
                                     #words_source])            
            map_fn_components.extend([phones_map_fn])#, phonemes_map_fn,
                                     #words_map_fn])
            batch_components.extend([None])#, None, None])

        space = CompositeSpace(space_components)
        source = tuple(source_components)
        self.data_specs = (space, source)
        self.map_functions = tuple(map_fn_components)
        self.batch_buffers = batch_components

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

    def _fetch_index(self, indices):
        if self.use_examples == None:
            digit = numpy.digitize(indices, self.cumulative_example_indexes) - 1
            return zip(digit,
                       numpy.array(indices) - self.cumulative_example_indexes[digit])
        else:
            return self.use_examples[ indices ]            

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

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indices):
        """
        .. todo::

            WRITEME
        """
        if type(indices) is slice:
            indices = numpy.arange(indices.start, indices.stop)
        self._validate_source(source)
        rval = []
        for so in source:                       
            batch_buffer = self.batch_buffers[self.data_specs[1].index(so)]
            dim = self.data_specs[0].components[self.data_specs[1].index(so)].dim
            if batch_buffer is None or batch_buffer.shape != (len(batch), dim):
                batch_buffer = numpy.zeros((len(indices), dim),
                                           dtype=self.data_specs[0].components[ self.data_specs[1].index(so) ].dtype)
            self.map_functions[ self.data_specs[1].index(so) ](indices, batch_buffer)
            rval.append(batch_buffer)
        return tuple(rval)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs                

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        
        if self.num_examples>10**6: # ShuffledSequentialSubsetIterator is slow and memory inefficient with many examples
            mode = RandomUniformSubsetIterator
            
        base_iterator = mode(self.num_examples, batch_size,
                                          num_batches, rng)        
        fdi = FiniteDatasetIterator(self,
                                     base_iterator,
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)
        return fdi


#class TIMITSequences(Dataset):
    #"""
    #Sequence-based TIMIT dataset
    #"""
    #_default_seed = (17, 2, 946)

    ## Mean and standard deviation of the acoustic samples from the whole
    ## dataset (train, valid, test).
    #_mean = 0.0035805809921434142
    #_std = 542.48824133746177

    #def __init__(self, which_set, frame_length, start=0, stop=None,
                 #audio_only=False, rng=_default_seed):
        #"""
        #Parameters
        #----------
        #which_set : str
            #Either "train", "valid" or "test"
        #frame_length : int
            #Number of acoustic samples contained in the sliding window
        #start : int, optional
            #Starting index of the sequences to use. Defaults to 0.
        #stop : int, optional
            #Ending index of the sequences to use. Defaults to `None`, meaning
            #sequences are selected all the way to the end of the array.
        #audio_only : bool, optional
            #Whether to load only the raw audio and no auxiliary information.
            #Defaults to `False`.
        #rng : object, optional
            #A random number generator used for picking random indices into the
            #design matrix when choosing minibatches.
        #"""
        #self.frame_length = frame_length
        #self.audio_only = audio_only

        ## RNG initialization
        #if hasattr(rng, 'random_integers'):
            #self.rng = rng
        #else:
            #self.rng = numpy.random.RandomState(rng)

        ## Load data from disk
        #self._load_data(which_set)
        ## Standardize data
        #for i, sequence in enumerate(self.raw_wav):
            #self.raw_wav[i] = (sequence - TIMIT._mean) / TIMIT._std

        ## Slice data
        #if stop is not None:
            #self.raw_wav = self.raw_wav[start:stop]
            #if not self.audio_only:
                #self.phones = self.phones[start:stop]
                #self.phonemes = self.phonemes[start:stop]
                #self.words = self.words[start:stop]
        #else:
            #self.raw_wav = self.raw_wav[start:]
            #if not self.audio_only:
                #self.phones = self.phones[start:]
                #self.phonemes = self.phonemes[start:]
                #self.words = self.words[start:]

        #samples_sequences = []
        #targets_sequences = []
        #phones_sequences = []
        #phonemes_sequences = []
        #words_sequences = []
        #for sequence_id, samples_sequence in enumerate(self.raw_wav):
            ## Sequence segmentation
            #samples_segmented_sequence = segment_axis(samples_sequence,
                                                      #frame_length,
                                                      #frame_length - 1)[:-1]
            #samples_sequences.append(samples_segmented_sequence)
            #targets_sequences.append(samples_sequence[frame_length:].reshape(
                #(samples_sequence[frame_length:].shape[0], 1)
            #))
            #if not self.audio_only:
                #target_phones = self.phones[sequence_id][frame_length:]
                #phones_sequences.append(target_phones.reshape(
                    #(target_phones.shape[0], 1)
                #))
                #target_phonemes = self.phonemes[sequence_id][frame_length:]
                #phonemes_sequences.append(target_phonemes.reshape(
                    #(target_phonemes.shape[0], 1)
                #))
                #target_words = self.words[sequence_id][frame_length:]
                #words_sequences.append(target_words.reshape(
                    #(target_words.shape[0], 1)
                #))

        #del self.raw_wav
        #self.samples_sequences = samples_sequences
        #self.targets_sequences = targets_sequences
        #self.data = [samples_sequences, targets_sequences]
        #if not self.audio_only:
            #del self.phones
            #del self.phonemes
            #del self.words
            #self.phones_sequences = phones_sequences
            #self.phonemes_sequences = phonemes_sequences
            #self.words_sequences = words_sequences
            #self.data.extend([phones_sequences, phonemes_sequences,
                              #words_sequences])
        #self.num_examples = len(samples_sequences)

        ## DataSpecs
        #features_space = VectorSequenceSpace(window_dim=self.frame_length)
        #features_source = 'features'

        #targets_space = VectorSequenceSpace(window_dim=1)
        #targets_source = 'targets'

        #space_components = [features_space, targets_space]
        #source_components = [features_source, targets_source]
        #batch_components = [None, None]

        #if not self.audio_only:
            #num_phones = numpy.max([numpy.max(sequence) for sequence
                                    #in self.phones_sequences]) + 1
            #phones_space = IndexSequenceSpace(
                #max_labels=num_phones,
                #window_dim=1,
                #dtype=str(self.phones_sequences[0].dtype)
            #)
            #phones_source = 'phones'

            #num_phonemes = numpy.max([numpy.max(sequence) for sequence
                                      #in self.phonemes_sequences]) + 1
            #phonemes_space = IndexSequenceSpace(
                #max_labels=num_phonemes,
                #window_dim=1,
                #dtype=str(self.phonemes_sequences[0].dtype)
            #)
            #phonemes_source = 'phonemes'

            #num_words = numpy.max([numpy.max(sequence) for sequence
                                   #in self.words_sequences]) + 1
            #words_space = IndexSequenceSpace(
                #max_labels=num_words,
                #window_dim=1,
                #dtype=str(self.words_sequences[0].dtype)
            #)
            #words_source = 'words'

            #space_components.extend([phones_space, phonemes_space,
                                     #words_space])
            #source_components.extend([phones_source, phonemes_source,
                                     #words_source])
            #batch_components.extend([None, None, None])

        #space = CompositeSpace(space_components)
        #source = tuple(source_components)
        #self.data_specs = (space, source)
        #self.batch_buffers = batch_components

        ## Defaults for iterators
        #self._iter_mode = resolve_iterator_class('shuffled_sequential')
        #self._iter_data_specs = (CompositeSpace((features_space,
                                                 #targets_space)),
                                 #(features_source, targets_source))

    #def _fetch_index(self, indexes):
        #digit = numpy.digitize(indexes, self.cumulative_example_indexes) - 1
        #return zip(digit,
                   #numpy.array(indexes) - self.cumulative_example_indexes[digit])

    #def _load_data(self, which_set):
        #"""
        #Load the TIMIT data from disk.

        #Parameters
        #----------
        #which_set : str
            #Subset of the dataset to use (either "train", "valid" or "test")
        #"""
        ## Check which_set
        #if which_set not in ['train', 'valid', 'test']:
            #raise ValueError(which_set + " is not a recognized value. " +
                             #"Valid values are ['train', 'valid', 'test'].")

        ## Create file paths
        #timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       #"timit/readable")
        #speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        #phonemes_list_path = os.path.join(timit_base_path,
                                          #"reduced_phonemes.pkl")
        #words_list_path = os.path.join(timit_base_path, "words.pkl")
        #speaker_features_list_path = os.path.join(timit_base_path,
                                                  #"spkr_feature_names.pkl")
        #speaker_id_list_path = os.path.join(timit_base_path,
                                            #"speakers_ids.pkl")
        #raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        #phonemes_path = os.path.join(timit_base_path,
                                     #which_set + "_x_phonemes.npy")
        #phones_path = os.path.join(timit_base_path,
                                     #which_set + "_x_phones.npy")
        #words_path = os.path.join(timit_base_path, which_set + "_x_words.npy")
        #speaker_path = os.path.join(timit_base_path,
                                    #which_set + "_spkr.npy")

        ## Load data. For now most of it is not used, as only the acoustic
        ## samples are provided, but this is bound to change eventually.
        ## Global data
        #if not self.audio_only:
            #self.speaker_info_list = serial.load(
                #speaker_info_list_path
            #).tolist().toarray()
            #self.speaker_id_list = serial.load(speaker_id_list_path)
            #self.speaker_features_list = serial.load(speaker_features_list_path)
            #self.words_list = serial.load(words_list_path)
            #self.phonemes_list = serial.load(phonemes_list_path)
        ## Set-related data
        #self.raw_wav = serial.load(raw_wav_path)
        #if not self.audio_only:
            #self.phonemes = serial.load(phonemes_path)
            #self.phones = serial.load(phones_path)
            #self.words = serial.load(words_path)
            #self.speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

    #def _validate_source(self, source):
        #"""
        #Verify that all sources in the source tuple are provided by the
        #dataset. Raise an error if some requested source is not available.

        #Parameters
        #----------
        #source : `tuple` of `str`
            #Requested sources
        #"""
        #for s in source:
            #try:
                #self.data_specs[1].index(s)
            #except ValueError:
                #raise ValueError("the requested source named '" + s + "' " +
                                 #"is not provided by the dataset")

    #def get_data_specs(self):
        #"""
        #Returns the data_specs specifying how the data is internally stored.

        #This is the format the data returned by `self.get_data()` will be.

        #.. note::

            #Once again, this is very hacky, as the data is not stored that way
            #internally. However, the data that's returned by `TIMIT.get()`
            #_does_ respect those data specs.
        #"""
        #return self.data_specs

    #def get(self, source, indexes):
        #"""
        #.. todo::

            #WRITEME
        #"""
        #if type(indexes) is slice:
            #indexes = numpy.arange(indexes.start, indexes.stop)
        #assert indexes.shape == (1,)
        #self._validate_source(source)
        #rval = []
        #for so in source:
            #rval.append(
                #self.data[self.data_specs[1].index(so)][indexes]
            #)
        #return tuple(rval)

    #@functools.wraps(Dataset.iterator)
    #def iterator(self, mode=None, batch_size=None, num_batches=None,
                 #rng=None, data_specs=None, return_tuple=False):
        #"""
        #.. todo::

            #WRITEME
        #"""
        #if data_specs is None:
            #data_specs = self._iter_data_specs

        ## If there is a view_converter, we have to use it to convert
        ## the stored data for "features" into one that the iterator
        ## can return.
        #space, source = data_specs
        #if isinstance(space, CompositeSpace):
            #sub_spaces = space.components
            #sub_sources = source
        #else:
            #sub_spaces = (space,)
            #sub_sources = (source,)

        #convert = []
        #for sp, src in safe_zip(sub_spaces, sub_sources):
            #convert.append(None)

        ## TODO: Refactor
        #if mode is None:
            #if hasattr(self, '_iter_subset_class'):
                #mode = self._iter_subset_class
            #else:
                #raise ValueError('iteration mode not provided and no default '
                                 #'mode set for %s' % str(self))
        #else:
            #mode = resolve_iterator_class(mode)

        #if batch_size is None:
            #batch_size = getattr(self, '_iter_batch_size', None)
        #if num_batches is None:
            #num_batches = getattr(self, '_iter_num_batches', None)
        #if rng is None and mode.stochastic:
            #rng = self.rng
        #return FiniteDatasetIterator(self,
                                     #mode(self.num_examples, batch_size,
                                          #num_batches, rng),
                                     #data_specs=data_specs,
                                     #return_tuple=return_tuple,
                                     #convert=convert)


if __name__ == "__main__":
    valid_timit = TIMITSequences("valid", frame_length=100, audio_only=False)
    data_specs = (CompositeSpace([VectorSequenceSpace(window_dim=100),
                                  VectorSequenceSpace(window_dim=1),
                                  VectorSequenceSpace(window_dim=62)]),
                  ('features', 'targets', 'phones'))
    it = valid_timit.iterator(mode='sequential', data_specs=data_specs,
                              num_batches=10, batch_size=1)
    for rval in it:
        print [val.shape for val in rval]
