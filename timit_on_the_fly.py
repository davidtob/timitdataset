from timit_raw_data import TIMITRawData
from pylearn2.datasets import Dataset


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
                 #overlap=0,
                 frames_per_example=1, # Important mostly when using fourier representation, in which case this is the window size
                 output_frames_per_example=1,
                 start=0,
                 stop=None,
                 audio_only=False,
                 representation='time', # or 'freq'
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
        self.timit_data = TIMITData( which_set, start, stop, audio_only, speaker_filter, phone_filter )
        
        self.frame_length = 1#frame_length
        #self.overlap = overlap
        self.frames_per_example = frames_per_example
        self.output_frames_per_example = output_frames_per_example
        self.offset = self.frame_length# - self.overlap
        self.audio_only = audio_only
        self.noise = noise
        self.noise_decay = noise_decay
        #self.speaker_filter = speaker_filter
        #self.phone_filter = phone_filter
        self.mid_third = mid_third
        self.use_examples = None
        #self.start = start
        #self.stop = stop
        self.representation = representation
        
        assert not self.representation=='freq' and self.audio_only==False # Phone map not implemented yet for Fourier data

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng 
        else:
            self.rng = numpy.random.RandomState(rng)
        

        if self.domain=='freq':
            self.compute_stft() 
        
        # Offset data for mapping example index to example
        examples_per_sequence = [0] + map( lambda x: len(x) - self.frames_per_example - self.output_frames_per_example + 1, self.raw_wav )
        self.cumulative_example_indexes = numpy.cumsum(examples_per_sequence)        
        self.num_examples = self.cumulative_example_indexes[-1]
        
        ## If requested, make further split of disk training set (only works well if the number of examples is small)
        #if which_set =='train_train' or which_set=='train_valid':
        #    digit = numpy.digitize(range(self.num_examples), self.cumulative_example_indexes) - 1
        #    ex_indices = zip(digit, numpy.array(range(self.num_examples)) - self.cumulative_example_indexes[digit])
        #    numpy.random.shuffle( ex_indices )
        #    if which_set == 'train_train':
        #        self.use_indices = ex_indices[:int(self.num_examples*0.8)]
        #    elif which_set=='train_valid':
        #        self.use_indices = ex_indices[int(self.num_examples*0.8):]
        #    self.num_examples = len(self.use_indices)

        print "number of examples", self.num_examples
            
        self.samples_sequences = self.data.raw_wav

        # DataSpecs
        features_space = VectorSpace( dim=self.frame_length * self.frames_per_example )
        features_source = 'features'
        targets_space = VectorSpace( dim=self.frame_length*self.output_frames_per_example )
        targets_source = 'targets'
        
        # Functions for fetching the X of an example
        def features_map_fn(indices, batch_buffer):
            for i, (sequence_index, example_index) in enumerate(self._fetch_index(indices)):
                batch_buffer[i,:] = self.samples_sequences[sequence_index][example_index:example_index+self.frames_per_example].ravel()
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
        
        # Functions for fetching the y of an xample
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
            phones_space = IndexSpace(max_labels=num_phones, dim=1, dtype=str(self.phone_nums[0].dtype))
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
        self._iter_data_specs = (CompositeSpace((features_space, targets_space)),
                                 (features_source, targets_source))

    def _fetch_index(self, indices):
        if self.use_examples == None:
            digit = numpy.digitize(indices, self.cumulative_example_indexes) - 1
            return zip(digit,
                       numpy.array(indices) - self.cumulative_example_indexes[digit])
        else:
            return self.use_examples[ indices ]            

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
