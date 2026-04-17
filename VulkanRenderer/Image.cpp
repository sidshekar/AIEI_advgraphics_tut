#include "Image.hpp"

using Gfx::Image;

Image::Image(vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::raii::ImageView&& imageView, vk::Extent3D extent, vk::Format format):
    m_image(std::move(image)),
    m_bufferMemory(std::move(bufferMemory)),
	m_imageView(std::move(imageView)),
    m_extent(extent),
    m_format(format)
{
}