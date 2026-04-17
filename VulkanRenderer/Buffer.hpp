#pragma once

#include "RHI.hpp"

namespace Gfx
{
	class Buffer
	{
    private:
        friend class RHI;

        Buffer(vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory, vk::DeviceSize size);

    public:
        Buffer(nullptr_t):
			m_buffer(nullptr), 
            m_bufferMemory(nullptr), 
            m_size(0)
        {}

        Buffer() = delete;

        operator vk::Buffer() const { return *m_buffer; }
        vk::Buffer operator*() const { return *m_buffer; }

        void map();
        void unmap();
		void* getMappedData() const { return m_mappedData; }

    private:
        vk::raii::Buffer m_buffer;
        vk::raii::DeviceMemory m_bufferMemory;
        vk::DeviceSize m_size;
		void* m_mappedData = nullptr;
    };
}