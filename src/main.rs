use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
	AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
	SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::sync::Arc;

fn main() {
	// The extensions we need to enable on the vulkan device.
	// We start with the extensions required by vulkano_win to create a window.
	let vk_required_extensions = vulkano_win::required_extensions();

	// create instance of vulkano
	let vk_instance = Instance::new(None, &vk_required_extensions, None).unwrap();

	// todo pick the best device here

	// pick the first device
	let vk_physical_device = PhysicalDevice::enumerate(&vk_instance).next().unwrap();

	println!(
		"Using device: {} (type: {:?})",
		vk_physical_device.name(),
		vk_physical_device.ty()
	);

	let event_loop = EventLoop::new();

	// Create our window and link vulkan to it.
	// This gives us a swapchain now.
	let surface = WindowBuilder::new()
		.build_vk_surface(&event_loop, vk_instance.clone())
		.unwrap();

	// todo add more queues for running commands in parallel (draw, compute, etc)

	// pick device queue for drawing
	let vk_queue_family = vk_physical_device
		.queue_families()
		.find(|&q| {
			// pick the first one that supports drawing the window
			q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
		})
		.unwrap();

	// vulkan device extension requirements
	let vk_device_ext = DeviceExtensions {
		khr_swapchain: true,
		..DeviceExtensions::none()
	};

	// create the vulkan device
	let (vk_device, mut vk_queues) = Device::new(
		vk_physical_device,
		vk_physical_device.supported_features(),
		&vk_device_ext,
		[(vk_queue_family, 0.5)].iter().cloned(),
	)
	.unwrap();

	// todo handle multiple queues when they are added
	// only use the first queue for now
	let vk_queue = vk_queues.next().unwrap();

	let (mut vk_swapchain, vk_images) = {
		let caps = surface.capabilities(vk_physical_device).unwrap();

		let alpha = caps.supported_composite_alpha.iter().next().unwrap();

		let format = caps.supported_formats[0].0;

		let dimensions: [u32; 2] = surface.window().inner_size().into();

		Swapchain::new(
			vk_device.clone(),
			surface.clone(),
			caps.min_image_count,
			format,
			dimensions,
			1,
			ImageUsage::color_attachment(),
			&vk_queue,
			SurfaceTransform::Identity,
			alpha,
			PresentMode::Fifo,
			FullscreenExclusive::Default,
			true,
			ColorSpace::SrgbNonLinear,
		)
		.unwrap()
	};

	// buffer for storing the vertices of the triangle
	let vertex_buffer = {
		#[derive(Default, Debug, Clone)]
		struct Vertex {
			position: [f32; 2],
		}
		vulkano::impl_vertex!(Vertex, position);

		CpuAccessibleBuffer::from_iter(
			vk_device.clone(),
			BufferUsage::all(),
			false,
			[
				Vertex {
					position: [-0.5, -0.25],
				},
				Vertex {
					position: [0.0, 0.5],
				},
				Vertex {
					position: [0.25, -0.1],
				},
			]
			.iter()
			.cloned(),
		)
		.unwrap();
	};

	mod vs {
		vulkano_shaders::shader! {
			ty: "vertex",
			src: "
				#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
		}
	}

	mod fs {
		vulkano_shaders::shader! {
			ty: "fragment",
			src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.0, 0.0, 0.0, 1.0);
				}
			"
		}
	}

	let vs = vs::Shader::load(vk_device.clone()).unwrap();
	let fs = fs::Shader::load(vk_device.clone()).unwrap();

	let render_pass = Arc::new(
		vulkano::single_pass_renderpass!(
			vk_device.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: vk_swapchain.format(),
					samples: 1,
				}
			},
			pass: {
				color: [color],
				depth_stencil: {}
			}
		)
		.unwrap(),
	);

	let pipeline = Arc::new(
		GraphicsPipeline::start()
			.vertex_input_single_buffer()
			.vertex_shader(vs.main_entry_point(), ())
			.triangle_list()
			.viewports_dynamic_scissors_irrelevant(1)
			.fragment_shader(fs.main_entry_point(), ())
			.render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
			.build(vk_device.clone())
			.unwrap(),
	);

	let mut dynamic_state = DynamicState {
		line_width: None,
		viewports: None,
		scissors: None,
		compare_mask: None,
		write_mask: None,
		reference: None,
	};

	let mut framebuffers =
		window_size_dependent_setup(&vk_images, render_pass.clone(), &mut dynamic_state);

	let mut recreate_swapchain = false;

	let mut previous_frame_end = Some(sync::now(vk_device.clone()).boxed());

	event_loop.run(move |event, _, control_flow| match event {
		Event::WindowEvent {
			event: WindowEvent::CloseRequested,
			..
		} => {
			*control_flow = ControlFlow::Exit;
		}
		Event::WindowEvent {
			event: WindowEvent::Resized(_),
			..
		} => {
			recreate_swapchain = true;
		}
		Event::RedrawEventsCleared => {
			previous_frame_end.as_mut().unwrap().cleanup_finished();

			if recreate_swapchain {
				let dimensions: [u32; 2] = surface.window().inner_size().into();
				let (new_swapchain, new_images) =
					match vk_swapchain.recreate_with_dimensions(dimensions) {
						Ok(r) => r,
						Err(SwapchainCreationError::UnsupportedDimensions) => return,
						Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
					};
				vk_swapchain = new_swapchain;
				framebuffers = window_size_dependent_setup(
					&new_images,
					render_pass.clone(),
					&mut dynamic_state,
				);
				recreate_swapchain = false;
			}

			let (image_num, suboptimal, acquire_future) =
				match swapchain::acquire_next_image(vk_swapchain.clone(), None) {
					Ok(r) => r,
					Err(AcquireError::OutOfDate) => {
						recreate_swapchain = true;
						return;
					}
					Err(e) => panic!("Failed to acquire next image: {:?}", e),
				};

			if suboptimal {
				recreate_swapchain = true;
			}

			let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

			let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
				vk_device.clone(),
				vk_queue.family(),
			)
			.unwrap();

			builder
				.begin_render_pass(
					framebuffers[image_num].clone(),
					SubpassContents::Inline,
					clear_values,
				)
				.unwrap()
				.draw(
					pipeline.clone(),
					&dynamic_state,
					vertex_buffer.clone(),
					(),
					(),
					vec![],
				)
				.unwrap()
				.end_render_pass()
				.unwrap();

			let command_buffer = builder.build().unwrap();

			let future = previous_frame_end
				.take()
				.unwrap()
				.join(acquire_future)
				.then_execute(vk_queue.clone(), command_buffer)
				.unwrap()
				.then_swapchain_present(vk_queue.clone(), vk_swapchain.clone(), image_num)
				.then_signal_fence_and_flush();

			match future {
				Ok(future) => {
					previous_frame_end = Some(future.boxed());
				}
				Err(FlushError::OutOfDate) => {
					recreate_swapchain = true;
					previous_frame_end = Some(sync::now(vk_device.clone()).boxed());
				}
				Err(e) => {
					println!("Failed to flush future: {:?}", e);
					previous_frame_end = Some(sync::now(vk_device.clone()).boxed());
				}
			}
		}
		_ => (),
	});
}

fn window_size_dependent_setup(
	images: &[Arc<SwapchainImage<Window>>],
	render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
	dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
	let dimensions = images[0].dimensions();

	let viewport = Viewport {
		origin: [0.0, 0.0],
		dimensions: [dimensions[0] as f32, dimensions[1] as f32],
		depth_range: 0.0..1.0,
	};
	dynamic_state.viewports = Some(vec![viewport]);

	images
		.iter()
		.map(|image| {
			let view = ImageView::new(image.clone()).unwrap();

			Arc::new(
				Framebuffer::start(render_pass.clone())
					.add(view)
					.unwrap()
					.build()
					.unwrap(),
			) as Arc<dyn FramebufferAbstract + Send + Sync>
		})
		.collect::<Vec<_>>()
}
