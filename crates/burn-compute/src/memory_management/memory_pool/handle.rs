use crate::memory_id_type;
use crate::memory_management::{MemoryBinding, MemoryHandle};

// The ChunkId allows to keep track of how many references there are to a specific chunk.
memory_id_type!(ChunkId, ChunkHandle);
// The SliceId allows to keep track of how many references there are to a specific slice.
memory_id_type!(SliceId, SliceHandle, SliceBinding);

/// A tensor memory handle, referring to either a chunk or a slice.
#[derive(Debug, Clone)]
pub enum MemoryPoolHandle {
    Mapped(SliceHandle),
    UnMapped { handle: SliceHandle, size: usize },
}

/// Binding of the [dynamic handle](DynamicHandle).
#[derive(Debug, Clone)]
pub enum MemoryPoolBinding {
    Mapped(SliceBinding),
    UnMapped { binding: SliceBinding, size: usize },
}

impl MemoryBinding for MemoryPoolBinding {
    fn is_mapped(&self) -> bool {
        match self {
            MemoryPoolBinding::Mapped(_) => true,
            MemoryPoolBinding::UnMapped {
                binding: _,
                size: _,
            } => false,
        }
    }
}

impl MemoryHandle<MemoryPoolBinding> for MemoryPoolHandle {
    fn new_unmapped(size: usize) -> Self {
        Self::UnMapped {
            handle: SliceHandle::new(),
            size,
        }
    }

    fn can_mut(&self) -> bool {
        match self {
            MemoryPoolHandle::Mapped(slice) => slice.can_mut(),
            MemoryPoolHandle::UnMapped { handle: _, size: _ } => false,
        }
    }

    fn binding(self) -> MemoryPoolBinding {
        match self {
            MemoryPoolHandle::Mapped(handle) => MemoryPoolBinding::Mapped(handle.binding()),
            MemoryPoolHandle::UnMapped { handle, size } => MemoryPoolBinding::UnMapped {
                binding: handle.binding(),
                size,
            },
        }
    }
}
