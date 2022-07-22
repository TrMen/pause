use std::fmt::Display;

#[derive(Clone)]
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    pub max_size: usize,
    // Cursor always points at the next elem to be inserted
    // e.g. the oldest element in the buffer
    cursor: usize,
    stage: Option<T>,
}

impl<T: Display> Display for CircularBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stage_str = self
            .stage
            .as_ref()
            .map(|s| format!("{}", s))
            .unwrap_or_else(|| "None".to_string());

        writeln!(f, "{{ ({}/{})", self.cursor, self.max_size)?;

        write!(f, "[")?;
        let mut offset = 0;
        // Print from oldest to newest computation
        // TODO: When arguments, get them printed in here somehow
        for i in self.cursor..self.buffer.len() {
            writeln!(f, "{offset}:\t{},", self.buffer[i])?;
            offset += 1;
        }
        for i in 0..self.cursor {
            writeln!(f, "{offset}:\t{},", self.buffer[i])?;
            offset += 1;
        }
        write!(f, "]")?;

        writeln!(f, "Stage: '{}' }}", stage_str)
    }
}

impl<T> CircularBuffer<T> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(size),
            max_size: size,
            cursor: 0,
            stage: None,
        }
    }

    pub fn newest(&self) -> Option<&T> {
        if self.cursor == 0 {
            self.buffer.get(self.buffer.len())
        } else {
            self.buffer.get(self.cursor - 1)
        }
    }

    pub fn oldest(&self) -> Option<&T> {
        if self.is_full() {
            Some(&self.buffer[self.cursor])
        } else {
            self.buffer.get(0)
        }
    }

    pub fn insert(&mut self, value: T) {
        // TODO: I'm sure there's a better way to do this
        if !self.is_full() {
            self.buffer.push(value);
            self.cursor = (self.cursor + 1) % self.max_size;
        } else {
            self.buffer[self.cursor] = value;
            self.cursor = (self.cursor + 1) % self.max_size;
        }
    }

    pub fn stage(&mut self, value: T) {
        self.stage = Some(value);
    }

    pub fn insert_stage(&mut self) {
        assert!(self.stage.is_some());
        let staged = self.stage.take().unwrap();
        self.insert(staged);
    }

    pub fn change_max_size(&mut self, new_size: usize) {
        self.buffer.reserve(new_size);
        self.max_size = new_size;
    }

    pub fn replace(&mut self, offset_from_oldest: usize, new: T) {
        assert!(offset_from_oldest <= self.buffer.len()); // Circular behavior is probably a bug in the caller
        let idx = (self.cursor + offset_from_oldest) % self.buffer.len();
        self.buffer[idx] = new;
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.max_size
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.buffer
            .iter()
            .skip(self.cursor)
            .chain(self.buffer.iter().take(self.cursor))
    }

    // TODO: Can make an endless circular buffer method (esp a mut one)
}
