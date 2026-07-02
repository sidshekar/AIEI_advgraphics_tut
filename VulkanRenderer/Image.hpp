#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
    struct ImageInfo
    {
        vk::ImageCreateInfo createInfo;
        std::vector<vk::Image> images;
        std::vector<vk::ImageView> imageViews;
    };

    class Image
    {
    private:
        friend class RHI;

        Image(
            const vk::ImageCreateInfo& createInfo, 
            std::vector<vk::raii::Image>&& images, 
            std::vector<vk::raii::DeviceMemory>&& imageMemories, 
            std::vector<vk::raii::ImageView>&& imageViews);

    public:
        Image(nullptr_t) {}

        Image() = delete;

        const vk::ImageCreateInfo& getCreateInfo() const { return m_createInfo; }
		const vk::ImageView& getImageView(int index) const { return *m_imageViews[index]; }
        const vk::Image& getImage(int index) const { return *m_images[index]; }
        int getImageCount() const { return static_cast<int>(m_images.size()); }

        const ImageInfo& getInfo() const { return m_info; }

    private:
        vk::ImageCreateInfo m_createInfo;
        std::vector<vk::raii::Image> m_images;
        std::vector<vk::raii::DeviceMemory> m_imageMemories;
        std::vector<vk::raii::ImageView> m_imageViews;
        ImageInfo m_info;
    };

    class Sampler
    {
    private:
        friend class RHI;

        Sampler(const vk::SamplerCreateInfo& createInfo, vk::raii::Sampler&& sampler);

    public:
        Sampler(nullptr_t) : m_sampler(nullptr) {}

        Sampler() = delete;

        const vk::SamplerCreateInfo& getCreateInfo() const { return m_createInfo; }
        const vk::Sampler& getSampler() const { return *m_sampler; }

    private:
        vk::SamplerCreateInfo m_createInfo;
        vk::raii::Sampler m_sampler;
    };
}