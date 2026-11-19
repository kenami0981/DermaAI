using Derma.Domain;
using FluentValidation;
using FluentValidation.Results;
using MediatR;
using Microsoft.AspNetCore.Identity;

namespace Derma.Application.Account
{
    public class AccountRegister
    {
        public class Command : IRequest<Guid>
        {
            public required string Email { get; set; }
            public required string UserName { get; set; }
            public required string Password { get; set; }
        }

        public class CommandValidator : AbstractValidator<Command>
        {
            public CommandValidator()
            {
                RuleFor(x => x.Email)
                    .NotEmpty().WithMessage("Email jest wymagany.")
                    .EmailAddress().WithMessage("Nieprawidłowy adres email.")
                    .MaximumLength(256);

                RuleFor(x => x.UserName)
                    .NotEmpty().WithMessage("Nazwa użytkownika jest wymagana.")
                    .MinimumLength(3).MaximumLength(50);

                RuleFor(x => x.Password)
                    .NotEmpty().WithMessage("Hasło jest wymagane.")
                    .MinimumLength(8).WithMessage("Hasło musi mieć co najmniej 8 znaków.");
            }
        }

        public class Handler : IRequestHandler<Command, Guid>
        {
            private readonly UserManager<AppUser> _userManager;

            public Handler(UserManager<AppUser> userManager)
            {
                _userManager = userManager;
            }

            public async Task<Guid> Handle(Command request, CancellationToken cancellationToken)
            {
                if (await _userManager.FindByEmailAsync(request.Email) != null)
                    throw new ValidationException(new[]
                    {
                        new ValidationFailure("Email", "Konto z tym adresem email już istnieje.")
                    });

                var user = new AppUser
                {
                    Id = Guid.NewGuid(),
                    Email = request.Email,
                    UserName = request.UserName
                };

                var result = await _userManager.CreateAsync(user, request.Password);

                if (!result.Succeeded)
                    throw new ValidationException(result.Errors
                        .Select(e => new ValidationFailure(e.Code, e.Description)));

                return user.Id;
            }
        }
    }
}