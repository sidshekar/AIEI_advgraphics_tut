#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
    struct BufferInfo
    {
        vk::BufferCreateInfo createInfo;
        std::vector<vk::Buffer> buffers;
    };

	class Buffer
	{
    private:
        friend class RHI;

        Buffer(
            const vk::BufferCreateInfo& createInfo, 
            std::vector<vk::raii::Buffer>&& buffers, 
            std::vector<vk::raii::DeviceMemory>&& bufferMemories);

    public:
        Buffer(nullptr_t) {}

        Buffer() = delete;

        const vk::BufferCreateInfo& getCreateInfo() const { return m_createInfo; }
        const vk::Buffer& getBuffer(int index) const { return *m_buffers[index]; }
        int getBufferCount() const { return static_cast<int>(m_buffers.size()); }

        void map();
        void unmap();
		void* getMappedData(int index) const { return m_mappedData[index]; }

        const BufferInfo& getInfo() const { return m_info; }

    private:
        vk::BufferCreateInfo m_createInfo;
        std::vector<vk::raii::Buffer> m_buffers;
        std::vector<vk::raii::DeviceMemory> m_bufferMemories;
        std::vector<void*> m_mappedData;
        BufferInfo m_info;
    };
}